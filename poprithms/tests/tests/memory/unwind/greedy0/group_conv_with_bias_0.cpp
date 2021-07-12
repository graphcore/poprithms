// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {
using namespace poprithms::memory::unwind;
using namespace poprithms::compute;

//  outer-act      outer-weight      outer-bias
//      |               |                |
//    slice           slice            slice (for i in {0,1,2})
//      |               |                |
//      +---------------+----------------+
//                      |
//                    call
//
//  where call(act, weight, bias) = conv(act, weight) + bias.
//
//
//  ... call ---+
//              |
//  ... call ---+--- concat --- output
//              |
//  ... call ---+
//
//  This for the 3 calls, one for each slice.
//
//

//  OpId  Name             OpType                         Shape
//  ----- ---------------- -----------------------------  -------
//  0     inner-act        Sink                           (10,8)
//  1     act-source       Source                         (10,8)
//  2     inner-weight     Sink                           (5,2)
//  3     weight-source    Source                         (5,2)
//  4     inner-bias       Sink                           (3)
//  5     conv             Barrier                        (5,3)
//  6     sumLike          SumLike(unwindIndex=0)         (5,3)
//  7     sumLike-barrier  Barrier                        (3)
//  8     outer-act        Sink                           (30,8)
//  9     outer-weight     Sink                           (15,2)
//  10    outer-bias       Sink                           (9)
//  11    act-slice(0)     SettSample((((10,20,0))()))    (10,8)
//  12    weight-slice(0)  SettSample((((5,10,0))()))     (5,2)
//  13    bias-slice(0)    SettSample((((3,6,0))))        (3)
//  14    call-out(0)      Sink                           (5,3)
//  15    act-slice(1)     SettSample((((10,20,10))()))   (10,8)
//  16    weight-slice(1)  SettSample((((5,10,5))()))     (5,2)
//  17    bias-slice(1)    SettSample((((3,6,3))))        (3)
//  18    call-out(1)      Sink                           (5,3)
//  19    act-slice(2)     SettSample((((10,20,20))()))   (10,8)
//  20    weight-slice(2)  SettSample((((5,10,10))()))    (5,2)
//  21    bias-slice(2)    SettSample((((3,6,6))))        (3)
//  22    call-out(2)      Sink                           (5,3)
//  23    concat           Concat(axis=1)                 (5,9)
//
//
//
//  we test that
//    outer-act has layouts from act-source
//    outer-weight has layouts from inner-weight
//    outer-bias has layouts from sumLike-barrier.
//    concat has layouts from conv.

void test0() {

  Graph g;

  const Shape actInShape{10, 8};
  const Shape weightShape{5, 2};
  const Shape actOutShape{5, 3};
  const Shape biasShape{3};

  const auto innerAct  = g.sink(actInShape);
  const auto actSource = g.source(actInShape);
  g.insertValuedPair(innerAct, actSource, 10.);

  const auto innerWeight  = g.sink(weightShape);
  const auto weightSource = g.source(weightShape);
  g.insertValuedPair(innerWeight, weightSource, 20.);

  g.setName(innerAct.opId(), "inner-act");

  // const auto actSource = innerActPair.source();
  g.setName(actSource.opId(), "act-source");

  // const auto innerWeight = innerWeightPair.sink();
  g.setName(innerWeight.opId(), "inner-weight");

  // const auto weightSource = innerWeightPair.source();
  g.setName(weightSource.opId(), "weight-source");

  const auto innerBias = g.sink(biasShape);
  g.setName(innerBias.opId(), "inner-bias");

  // We choose to represent the convolution as a barrier. In practise, as the
  // output probably does not depend on the input with the poplibs'
  // implementation (see T32143), it can be represented as a fixedPoint. This
  // would be advantagueous if an input, or any tensor preceding the
  // convolution in the DAG, would benefit from having a layout derived from
  // the output.
  const auto convOp = g.barrier({innerAct, innerWeight}, {actOutShape});
  g.setName(convOp, "conv");

  const auto sumLikeOut = g.sumLike({{convOp, 0}, innerBias}, InIndex(0), 5.);
  g.setName(sumLikeOut.out().opId(), "sumLike");

  if (sumLikeOut.mappings().size() != 1) {
    throw poprithms::test::error(
        "Expected 1 element in SumLikeMappings for this binary add.");
  }
  g.setName(sumLikeOut.barrier(0), "sumLike-barrier");

  const int64_t repl{3};
  const auto outerAct = g.sink(actInShape.broadcast(repl, 0));
  g.setName(outerAct.opId(), "outer-act");

  const auto outerWeight = g.sink(weightShape.broadcast(repl, 0));
  g.setName(outerWeight.opId(), "outer-weight");

  const auto outerBias = g.sink(biasShape.broadcast(repl, 0));
  g.setName(outerBias.opId(), "outer-bias");

  TensorIds callOuts;
  for (uint64_t i = 0; i < repl; ++i) {

    const auto act0 = actInShape.dim_u64(0);
    auto actSlice   = g.slice(outerAct, i * act0, (i + 1) * act0);
    g.setName(actSlice.opId(), "act-slice(" + std::to_string(i) + ")");

    const auto weight0 = weightShape.dim_u64(0);
    auto weightSlice   = g.slice(outerWeight, i * weight0, (i + 1) * weight0);
    g.setName(weightSlice.opId(), "weight-slice(" + std::to_string(i) + ")");

    const auto bias0 = biasShape.dim_u64(0);
    auto biasSlice   = g.slice(outerBias, i * bias0, (i + 1) * bias0);
    g.setName(biasSlice.opId(), "bias-slice(" + std::to_string(i) + ")");

    auto callOut = g.call({actSlice, weightSlice, biasSlice},
                          {innerAct, innerWeight, innerBias},
                          {sumLikeOut.out()},
                          11.)[0];
    callOuts.push_back(callOut);
    g.setName(callOut.opId(), "call-out(" + std::to_string(i) + ')');
  }

  auto out = g.concat(callOuts, 1);
  g.setName(out.opId(), "concat");

  const auto sAndBs = HostTensorHelper::arangeBarriers(g);
  const Solution soln(std::move(g));

  std::map<TensorId, host::Tensor> hosts;

  //  test 1) outer-act has layouts from act-source
  auto outerActTensor  = HostTensorHelper::get(soln, outerAct, sAndBs);
  auto actSourceTensor = HostTensorHelper::get(soln, actSource, sAndBs);
  auto actExpected     = host::Tensor::concat(
      {actSourceTensor, actSourceTensor, actSourceTensor}, 0);
  actExpected.assertAllEquivalent(outerActTensor);

  // test 2) outer-weight has layouts from inner-weight
  auto outerWeightTensor  = HostTensorHelper::get(soln, outerWeight, sAndBs);
  auto weightSourceTensor = HostTensorHelper::get(soln, weightSource, sAndBs);
  auto weightExpected     = host::Tensor::concat(
      {weightSourceTensor, weightSourceTensor, weightSourceTensor}, 0);
  weightExpected.assertAllEquivalent(outerWeightTensor);

  // test 3) outer-bias has layouts from sumLike-barrier.
  auto outerBiasTensor = HostTensorHelper::get(soln, outerBias, sAndBs);
  auto sumLikeBarrierTensor =
      HostTensorHelper::get(soln, sumLikeOut.reduced(0), sAndBs);
  auto biasExpected = host::Tensor::concat(
      {sumLikeBarrierTensor, sumLikeBarrierTensor, sumLikeBarrierTensor}, 0);
  outerBiasTensor.assertAllEquivalent(biasExpected);

  //  concat has layouts from conv.
  auto outTensor  = HostTensorHelper::get(soln, out, sAndBs);
  auto convTensor = HostTensorHelper::get(soln, {convOp, 0}, sAndBs);
  auto outExpected =
      host::Tensor::concat({convTensor, convTensor, convTensor}, 1);
  outExpected.assertAllEquivalent(outTensor);
}

} // namespace

int main() {
  test0();
  return 0;
}
