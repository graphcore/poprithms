// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <numeric>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/prune/pruner.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/testutil/finitedifference.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace {

using namespace poprithms::common::compute;
using poprithms::common::compute::Pruner;

void testNumericalTrainPrune0() {
  const uint64_t rf{1};
  SlickGraph m(1000, ReplicationFactor::create(rf));

  /**
   *  sg0:
   *
   *       +-----------+
   *       +           |
   * in0 --+           +---> copy out all
   *       |  (math)   |
   * in1 --+           |
   *       |           |
   *       +-----------+
   * */
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.variable(DType::Float32, {4, 3}, m.rootIpu());
  auto in1  = in0.variable();
  auto z    = in0 * in0 * in1;
  auto out0 = in0.copy();
  auto out1 = z.sin().cos();
  (void)out1;

  /**
   *  sg1 :
   *
   *  mainIn0---------------+
   *    |                   +-- sg0 - out0 --+
   *    +-------- mainIn1---+                +--- sg0 -- out0 -> loss
   *    +-------- mainIn2----------====------+                    .
   *                                                              .
   *               host <----------------- dMainIn0 <--------- autodiff
   *
   * */
  auto sg1         = m.createSubGraph("sg1");
  auto mainIn0host = sg1.hostFloat32Variable({1, 1, 4, 3});
  auto mainIn0     = mainIn0host.hostToIpu(m.rootIpu());
  auto mainIn1     = mainIn0.copy();
  auto mainIn2     = mainIn0.copy();
  auto c0 =
      sg1.call(sg0, {{mainIn0, in0}, {mainIn1, in1.id()}}, sg0.tensorIds());

  auto c1 = sg1.call(
      sg0, {{out0.dstInCaller(c0), in0}, {mainIn2, in1}}, sg0.tensorIds());
  auto loss = m.tensor(out0.dstInCaller(c1)).reduceSum(Shape{});

  Autodiffer ad(m);
  ad.backwardInGraph(
      {loss}, m.tensorIds(sg1), {mainIn0}, {loss.constant(1.)});

  // We have constructed the graphs so that this is all 1's.
  auto gradOfIn0 = m.tensor(ad.gradInfo(sg1).targetGradInGradGraph(mainIn0))
                       .ipuToHost(CircularBufferCount(1));

  auto lossOnHost = loss.ipuToHost(1);

  m.setRunnable({gradOfIn0.subGraphId()});

  Pruner::preserveHostTensors(m);

  SimExecutable se(m);

  auto hv = HostTensor::uniformFloat32(0.5, 1, {1, 1, 4, 3}, 1011);
  se.run(sg1);

  se.getHostValue(gradOfIn0).assertAllClose(
      HostTensor::float32(1).expand({1, 1, 4, 3}), 1e-6, 1e-6);

  poprithms::common::compute::testutil::finiteDifferenceTest<Tensor>(
      se,
      lossOnHost,
      mainIn0host,
      gradOfIn0,
      std::unordered_map<TensorId, HostTensor>{{mainIn0host, hv}},
      1011,
      1e-1, // perturbation. We can afford to have it very large, as the
            // gradient is 1 everywhere. with float32, it needs to be large to
            // offset rounding errors.
      1e-9, // epsilon0
      1e-4  // threshold. Largish for float32.
  );

  if (m.nSubGraphs() != 3) {
    throw poprithms::test::error(
        "Should be exactly 3 sub-graphs: sg0, sg0's gradient, and "
        "sg1 (which is a fwd-bwd graph)");
  }
}

void testNumericalPrune0() {
  SlickGraph m;
  auto sg0     = m.createSubGraph("sg0");
  auto in0     = sg0.variable(DType::Float32, {5}, m.rootIpu());
  auto out0    = in0.sin().abs();
  auto sg2     = m.createSubGraph("sg2");
  auto in1host = sg2.variable(DType::Float32, {1, 1, 5}, m.host());
  auto in1     = in1host.hostToIpu(m.rootIpu());
  auto c0      = sg2.call(sg0, {{in1.abs().sqrt(), in0}}, sg0.tensorIds());
  auto c1   = sg2.call(sg0, {{out0.dstInCaller(c0), in0}}, sg0.tensorIds());
  auto loss = out0.dstInCaller(c1).abs().reduceSum();
  Autodiffer ad(m);
  ad.backwardInGraph({loss}, sg2.tensorIds(), {in1}, {loss.constant(1)});
  auto finale = m.tensor(ad.gradInfo(sg2).targetGradInGradGraph(in1))
                    .ipuToHost(CircularBufferCount(1));
  m.setRunnable({finale.subGraphId()});

  auto lossOnHost = loss.ipuToHost(1);
  Pruner::preserveHostTensors(m);

  m.verifyValid();
  SimExecutable se(m);
  auto hv = HostTensor::uniformFloat32(0.5, 1, {1, 1, 5}, 1011);
  poprithms::common::compute::testutil::finiteDifferenceTest<Tensor>(
      se,
      lossOnHost,
      in1host,
      finale,
      std::unordered_map<TensorId, HostTensor>{{in1host, hv}},
      1011,
      1e-3, // perturbation.
      1e-9, // epsilon0
      1e-2  // threshold. Largish for float32.
  );
}

void testPrune() {
  SlickGraph m;
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat32Variable({1, 1, 5, 5}).hostToIpu(m.rootIpu());
  auto sg1  = m.createSubGraph("sg1");
  auto in1  = sg1.variable(DType::Float32, {5, 5}, m.rootIpu());
  auto out1 = in1.sin();
  auto out2 = in1.abs();
  (void)out2;
  auto c0 = sg0.call(sg1, {{in0, in1}}, {{out1.id()}});
  out1.dstInCaller(c0).relu().ipuToHost(CircularBufferCount(1));
  m.setRunnable({sg0});
  Pruner::preserveHostTensors(m);
  m.verifyValid();
  if (m.opIds<poprithms::common::compute::Abs>().size() != 0) {
    throw poprithms::test::error("Abs not on path to host");
  }
}

void testManualRecompute0() {

  int64_t N{40};

  auto initVal = HostTensor::uniformFloat64(-4, 4, {N}, 1011);

  // The non-recompute version:
  auto grad1 = [N, initVal]() {
    SlickGraph m;
    auto sg0 = m.createSubGraph("sg0");
    auto in0 = sg0.hostFloat64Variable({N});
    auto out = in0.sin().abs().cos().relu().reduceSum(Shape{});
    auto dIn = Autodiffer(m).backward(out, {in0})[0];
    m.setRunnable({sg0});
    SimExecutable cms(m);
    cms.setHostValue(in0, initVal);
    cms.run(sg0);
    return cms.getHostValue(dIn);
  }();

  // The recompute version:
  auto grad0 = [N, initVal]() {
    SlickGraph m;
    auto sg0 = m.createSubGraph("sg0");

    // Program 1 (sin + abs) which will be recompute.
    auto e0 = sg0.hostFloat64Variable({N});
    auto e1 = e0.sin().abs();

    Autodiffer ad(m);
    auto gg0 = ad.backwardOutOfGraph({e1}, sg0.tensorIds(), {e0});

    // Program 2 (cos + relu) which will to be recomputed.
    auto sg1 = m.createSubGraph("sg1");
    auto f0  = e0.variable(sg1);
    auto f1  = f0.cos().relu();
    auto gg1 = ad.backwardOutOfGraph({f1}, sg1.tensorIds(), {f0});

    // The main graph: call sg0, then sg1, then create a loss. The call to sg0
    // does not output all internal tensors.
    auto sg2      = m.createSubGraph("sg2");
    auto g0       = f0.variable(sg2);
    auto call0    = sg2.call(sg0, {{g0, e0}}, {{e1.id()}});
    auto call1    = sg2.callAllOut(sg1, {{e1.dstInCaller(call0), f0}});
    auto call1Out = f1.dstInCaller(call1);
    auto loss     = call1Out.reduceSum(Shape{});

    // Backwards to the output of call1.
    auto dCall1Out = ad.backward(loss, {call1Out})[0];

    auto getCpIns = [&ad, &m](const SubGraphId &gradGraph, OpId callId) {
      std::vector<std::pair<TensorId, TensorId>> copyIns;
      for (auto cpp : ad.gradInfo(gradGraph).checkpointPairs()) {
        copyIns.push_back({m.tensor(cpp.inNonGradGraph).dstInCaller(callId),
                           cpp.inGradGraph});
      }
      return copyIns;
    };
    // run the gradient of sg1:
    auto copyIns = getCpIns(gg1, call1);

    copyIns.push_back({dCall1Out, ad.gradInfo(gg1).gradInputInGradGraph(f1)});

    auto df0_in_gg1 = m.tensor(ad.gradInfo(gg1).targetGradInGradGraph(f0));

    auto gg1call = sg2.call(gg1, copyIns, {{df0_in_gg1}});
    auto df0     = df0_in_gg1.dstInCaller(gg1call);

    // re-run sg0, and get all internal tensors:
    auto call0_repeat = sg2.callAllOut(sg0, {{g0, e0}});

    // run gradient of sg0, using the recomputed tensors:
    copyIns = getCpIns(gg0, call0_repeat);

    copyIns.push_back({df0, ad.gradInfo(gg0).gradInputInGradGraph(e1)});

    auto gg0call = sg2.call(
        gg0, copyIns, {{ad.gradInfo(gg0).targetGradInGradGraph(e0)}});
    auto de0 = m.tensor(ad.gradInfo(gg0).targetGradInGradGraph(e0))
                   .dstInCaller(gg0call);

    m.setRunnable({sg2});
    SimExecutable cms(m);
    cms.setHostValue(g0, initVal);
    cms.run(sg2);

    return cms.getHostValue(de0);
  }();

  grad0.assertAllClose(grad1, 1e-4, 1e-4);
}

void testSeriouslyManual0() {

  SlickGraph m;
  // sg0. x -> |x|.
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({3});
  auto out0 = in0.abs();

  // sg1. x, y -> x * sign(y).
  auto sg1     = m.createSubGraph("sg1");
  auto gradIn  = in0.variable(sg1);
  auto cpIn    = in0.variable(sg1);
  auto gradOut = cpIn.signum() * gradIn;

  // sg2. call sg0.
  auto sg2   = m.createSubGraph("sg2");
  auto in2   = in0.variable(sg2);
  auto call0 = sg2.call(sg0, {{in2, in0}}, {out0, in0});

  // Manually set sg1 to be the gradient of sg0, and "connect the dots"
  auto gInfo = poprithms::autodiff::automatic::GradInfo::outOfGraph(
      sg0,
      sg1,
      {{out0, gradIn}}, // gradIn is the input gradient of out0.
      {{in0, cpIn}},    // in0 is the checkpoint input.
      {{in0, gradOut}}  // the gradient of in0 is gradOut.
  );

  // register the gradient relationship. Hereafter, if the op call0 will be
  // differentiated as a call to sg1.

  Autodiffer ad(m);
  ad.insertGradInfo(gInfo);
  ad.setGrad(call0, CalleeIndex(0), sg1);

  auto loss = out0.dstInCaller(call0).reduceSum(Shape({}));
  auto dIn2 = ad.backward(loss, {in2})[0];

  m.setRunnable({sg2});

  SimExecutable cms(m);

  cms.setHostValue<double>(in2, {-4, 5, -2});
  cms.run(sg2);
  cms.getHostValue(dIn2).assertAllEquivalent(
      HostTensor::float64({3}, {-1, +1, -1}));
}

void testResetOut0() {

  // where i is the tensor to use as the replacement when x1 is removed:
  auto test = [](int i) {
    SlickGraph m;

    auto sg0 = m.createSubGraph("sg0");
    auto x0  = sg0.variable(DType::Int16, {}, m.host());
    auto x1  = x0.variable();
    auto x2  = x0.variable();
    auto x3  = x0.variable(DType::Float32);

    auto sg1  = m.createSubGraph("sg1");
    auto call = sg1.call(sg0, {}, {x0, x1});
    (void)call;

    TensorId sub = i == 0 ? x0.id() : (i == 2 ? x2 : x3);

    if (i == 2) {
      m.removeOp(x1.opId(), {sub}, "can replace x1 with x2.");
      m.verifyValid();
      return;
    }

    int caught{0};
    try {
      m.removeOp(x1.opId(), {sub}, "can't duplicate copy out source");
      m.verifyValid();
    } catch (const poprithms::error::error &) {
      caught = true;
    }

    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error of of replacing with x" + std::to_string(i));
    }
  };

  test(0);
  test(2);
  test(3);
}

void testNoComputeInCall0() {

  std::cout << "\n\ntestNoComputeInCall0" << std::endl;

  SlickGraph g;
  auto caller  = g.createSubGraph("caller");
  auto xCaller = caller.hostFloat32Variable({});

  auto callee  = g.createSubGraph("callee");
  auto xCallee = callee.hostFloat32Variable({});

  auto callOp = caller.call(callee, {{{xCaller, xCallee}}}, {xCallee});
  auto loss   = (xCallee.dstInCaller(callOp));

  Autodiffer ad(g);
  auto dx = ad.backward(loss, {xCaller})[0];

  g.setRunnable({caller});

  SimExecutable se(g);
  se.setHostValue(xCaller, HostTensor::float32({}, {7.}));
  se.run(caller);

  std::cout << g << std::endl;

  se.getHostValue(dx).assertAllEquivalent(HostTensor::float32({}, {1.}));
}

} // namespace

int main() {
  testPrune();
  testResetOut0();
  testNumericalPrune0();
  testSeriouslyManual0();
  testManualRecompute0();
  testNumericalTrainPrune0();
  testNoComputeInCall0();
  return 0;
}
