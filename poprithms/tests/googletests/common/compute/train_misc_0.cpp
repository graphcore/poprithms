// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <gmock/gmock.h>
#include <sstream>

#include <poprithms/autodiff/testutil/finitedifference.hpp>
#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/scheduler.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {

using namespace poprithms::common::compute;
using Ad = Autodiffer<SlickGraph, AutomaticMutator>;

} // namespace

// This test checks that recomputation does happen when only the inputs to a
// graph are checkpointed.
TEST(CommonComputeTrainMisc0, Recompute0) {

  SlickGraph graph;

  /**
   * (1)    out = sqrt(sin(in) + 2).
   *
   * (2)    dOut = 1/(sqrt(sin(in) + 2)) * (1/2) * cos(in).
   * */
  auto sgFwd = graph.createSubGraph("fwd");
  auto d     = sgFwd.variable(DType::Float64, {2, 2}, graph.host()).name("d");
  auto c     = d.constant(2.0);
  auto out   = (d.sin() + c).sqrt().reduceSum(Shape({}));

  Ad ad(graph);
  auto sgBwdId = ad.backwardOutOfGraph(
      /* gradsProvidedFor = */ {out},
      /* checkpoints      = */ {d},
      /* requiresGrad     = */ {d});

  SubGraph sgBwd(sgBwdId, graph);

  auto &&gi = ad.gradInfo(sgBwd);

  // Expect the sin to be run for recomputation, too.
  EXPECT_EQ(graph.opIds<Sin>().size(), 2);

  graph.setRunnable({sgFwd, sgBwd});

  SimExecutable se(graph);

  // Compute the gradient of d0 using sbBwd:
  auto d0 = HostTensor::float64({2, 2}, {1, 2, 3, 4});
  se.setHostValue(gi.checkpointInGradGraph(d), d0);
  se.setHostValue(gi.gradInputInGradGraph(out), HostTensor::float64(1));
  se.run(sgBwd);
  auto g0 = se.getHostValue(gi.targetGradInGradGraph(d));

  // Perform finite-difference method to confirm the gradient is correct:
  auto fwd = [&](const HostTensor &ht) {
    se.setHostValue(d, ht);
    se.run(sgFwd);
    auto v = se.getHostValue(out).copy();
    return v;
  };
  double perturbationSize{0.001};
  uint64_t seed{1011};
  double eps0{1e-10};
  double threshold{1e-5};
  poprithms::autodiff::testutil::Checker::check(
      fwd, d0.copy(), g0, perturbationSize, seed, eps0, threshold);

  // We can also check the gradient against the derivation (2) above.
  auto expected = (d0.sin().add(2)).sqrt().pow(-1).mul(0.5).mul(d0.cos());
  g0.assertAllClose(expected, 1e-6, 1e-6, "compare to hand-derived gradient");
}

// Test that you can train through this inplace operation.
TEST(CommonComputeTrainMisc0, ThroughFill0) {
  SlickGraph m;
  SubGraph sg0 = m.createSubGraph("sg0");
  auto W       = sg0.hostFloat64Variable({4, 4});
  auto out     = W.fill_(HostTensor::float64(1.)).reduceSum(Shape({}));
  auto dW      = Ad(m).backward(out, {W})[0];
  m.setRunnable({sg0});
  SimExecutable cms(m);
  cms.setHostValue(W, HostTensor::uniformFloat64(-1, 1, {4, 4}, 1011));
  cms.run(sg0);
  // Expect the gradient of W to be entirely 0.
  EXPECT_EQ(cms.getHostValue(dW).allZero(), true);
}

TEST(CommonComputeTrainMisc0, ThroughAddInplace0) {
  SlickGraph m;
  SubGraph sg0 = m.createSubGraph("sg0");
  auto W       = sg0.hostFloat64Variable({3});
  auto out     = W.abs().add_(W.constant(1)).reduceSum(Shape({}));
  auto dW      = Ad(m).backward(out, {W})[0];
  m.setRunnable({sg0});
  SimExecutable cms(m);
  auto hostW = HostTensor::float64({3}, {1, -3, 2});
  cms.setHostValue(W, hostW.copy());
  cms.run(sg0);
  cms.getHostValue(dW).assertAllEquivalent(
      HostTensor::float64({3}, {+1, -1, +1}));
}

// Test that casting to an integer kills backprop.
TEST(CommonComputeTrainMisc0, ThroughCast0) {
  SlickGraph m;
  SubGraph sg0 = m.createSubGraph("sg0");
  auto W       = sg0.hostFloat64Variable({3});
  auto out0    = W.to(DType::Float32).mul(W.constant(DType::Float32, 7));
  auto out1 =
      W.to(DType::Int32).mul(W.constant(DType::Int32, 11)).to(DType::Float32);
  auto loss = (out0 + out1).reduceSum(Shape({}));
  auto dW   = Ad(m).backward(loss, {W})[0];
  m.setRunnable({sg0});
  SimExecutable cms(m);
  auto hostW = HostTensor::float64({3}, {1, 0, -1});
  cms.setHostValue(W, hostW.copy());
  cms.run(sg0);
  cms.getHostValue(dW).assertAllEquivalent(
      HostTensor::float64({3}, {+7, +7, +7}));
}

TEST(CommonComputeTrainMisc0, ThroughInv0) {
  SlickGraph m;
  SubGraph sg0 = m.createSubGraph("sg0");
  auto W       = sg0.hostFloat64Variable({3});
  auto loss    = (W.inv() - W.constant(1) / W).reduceSum(Shape{});
  auto dW      = Ad(m).backward(loss, {W})[0];
  m.setRunnable({sg0});
  SimExecutable cms(m);
  auto hostW = HostTensor::float64({3}, {1, 2, -0.5});
  cms.setHostValue(W, hostW.copy());
  cms.run(sg0);
  cms.getHostValue(dW).assertAllEquivalent(
      HostTensor::float64({3}, {0., 0, 0}));
}

TEST(CommonComputeTrainMisc0, ThroughMaxAndMin0) {
  enum class Extremum { Max = 0, Min };

  auto test = [](Extremum e) {
    SlickGraph m;
    SubGraph sg0 = m.createSubGraph("sg0");
    auto x       = sg0.hostFloat64Variable({3});
    auto y       = x.variable();

    auto hostX = HostTensor::float64({3}, {1, 0.1, -1});
    auto hostY = HostTensor::float64({3}, {-2, 0, 5});

    // 1, 0.1 5
    auto out0 = e == Extremum::Max ? x.max(y) : x.min(y);

    // 1, 0.1, 5 (y updated inplace).
    auto out1 = e == Extremum::Min ? y.min_(x) : y.max_(x);

    // in max case: loss = 2*x[0] + 2*x[1]+ 2*y[2]
    auto loss = (out0 + out1).reduceSum(Shape{});

    auto grads = Ad(m).backward(loss, {x, y});
    m.setRunnable({sg0});
    SimExecutable cms(m);
    cms.setHostValue(x, hostX);
    cms.setHostValue(y, hostY);
    cms.run(sg0);

    auto mask      = HostTensor::float64({3}, {1, 1, 0});
    auto negMask   = HostTensor::float64({3}, {0, 0, 1});
    auto expectedX = e == Extremum::Max ? mask.mul(2) : negMask.mul(2);
    auto expectedY = HostTensor::float64(2) - expectedX;

    auto dX = cms.getHostValue(grads[0]);
    dX.assertAllEquivalent(expectedX);

    auto dY = cms.getHostValue(grads[1]);
    dY.assertAllEquivalent(expectedY);
  };

  test(Extremum::Max);
  test(Extremum::Min);
}

// A chain of ops which together combine to be the identity: checks that
// identity is also identity.
TEST(CommonComputeTrainMisc0, CancelChain0) {
  SlickGraph m;
  SubGraph sg0 = m.createSubGraph("sg0");
  auto W       = sg0.hostFloat64Variable({10});
  auto out     = W.neg().neg();
  out          = out.relu() + out.neg().relu().neg();
  out          = out.abs().sqrt().pow(out.constant(2)) -
        out.neg().relu().mul(out.constant(2));
  out       = out.exp().log();
  auto loss = out.reduceSum(Shape({}));
  auto dW   = Ad(m).backward(loss, {W})[0];
  m.setRunnable({sg0});
  SimExecutable cms(m);
  auto hostW = HostTensor::uniformFloat64(-3, 3, {10}, 1011);
  cms.setHostValue(W, hostW.copy());
  cms.run(sg0);
  cms.getHostValue(dW).assertAllClose(
      HostTensor::float64(1).expand({10}), 1e-5, 1e-5);
}

TEST(CommonComputeTrainMisc0, ThroughDynamicSlice0) {
  SlickGraph m;
  SubGraph sg0   = m.createSubGraph("sg0");
  auto sliceable = sg0.hostFloat32Variable({6});
  int32_t nSlices{2};
  int64_t sliceSize{2};
  auto offset = sg0.variable(DType::Unsigned32, {nSlices, 1}, m.host());
  auto sliced =
      sliceable.dynamicMultiSlice(offset, Dimensions{0}, {sliceSize});
  auto loss       = (sliced * sliced).reduceSum(Shape({}));
  auto dSliceable = Ad(m).backward(loss, {sliceable})[0];
  m.setRunnable({sg0});
  SimExecutable cms(m);
  cms.setHostValue<float>(sliceable, {5, 6, 7, 8, 9, 10});
  cms.setHostValue<uint32_t>(offset, {4, 1});
  cms.run(sg0);
  cms.getHostValue(dSliceable)
      .assertAllEquivalent(HostTensor::float32({6}, {0, 12, 14, 0, 18, 20}));
}

TEST(CommonComputeTrainMisc0, ThroughDynamicMax0) {
  SlickGraph m;
  SubGraph sg0 = m.createSubGraph("sg0");
  int64_t M{3};
  int64_t N{4};
  int64_t S{2};

  auto sliceable = sg0.hostFloat32Variable({M, S});
  auto slice     = sliceable.variable({N, S});
  auto offset    = sg0.variable(DType::Unsigned32, {N}, m.host());
  auto updated   = sliceable.dynamicMultiUpdateMax_(slice, offset);
  auto loss      = updated.pow(slice.constant(2)).reduceSum(Shape({}));

  auto dSlice = Ad(m).backward(loss, {slice})[0];

  m.setRunnable({sg0});

  SimExecutable se(m);

  //  -3  -2
  //  -1   1
  //   2   3
  auto hSliceable =
      HostTensor::float32(sliceable.shape(), {-3, -2, -1, 1, 2, 3});

  // slice:
  //        offsets:
  //  1  4    2
  //  1 -1    1
  //  5  2    2
  // -5 -5    0

  auto hSlice =
      HostTensor::float32(slice.shape(), {1, 4, 1, -1, 5, 2, -5, -5});

  auto hOffset = HostTensor::unsigned32(offset.shape(), {2, 1, 2, 0});

  se.setHostValue(sliceable, hSliceable);
  se.setHostValue(slice, hSlice);
  se.setHostValue(offset, hOffset);

  se.run(sg0);

  se.getHostValue(updated).assertAllEquivalent(
      HostTensor::float32(sliceable.shape(), {-3, -2, 1, 1, 5, 4}));

  se.getHostValue(dSlice).assertAllEquivalent(
      HostTensor::float32(slice.shape(), {0, 8, 2, 0, 10, 0, 0, 0}));
}

TEST(CommonComputeTrainMisc0, ThroughDynamciUpdate0) {

  SlickGraph m;
  SubGraph sg0   = m.createSubGraph("sg0");
  auto sliceable = sg0.hostFloat32Variable({6});
  int32_t nSlices{2};
  int64_t sliceSize{2};
  auto offset = sg0.variable(DType::Unsigned32, {nSlices, 1}, m.host());
  auto sliced = sliceable.variable({nSlices, sliceSize});

  auto loss = sliceable.dynamicMultiUpdate_(sliced, offset, Dimensions{0})
                  .pow(sliceable.constant(2))
                  .reduceSum(Shape({}));

  TensorIds targs{sliced, sliceable};
  auto grads      = Ad(m).backward(loss, targs);
  auto dSlice     = grads[0];
  auto dSliceable = grads[1];

  m.setRunnable({sg0});
  SimExecutable cms(m);
  cms.setHostValue<float>(sliceable, {5, 6, 7, 8, 9, 10});
  cms.setHostValue<uint32_t>(offset, {4, 1});

  HostTensor vSliced = HostTensor::float32({2, 2}, {1, 2, 3, 4});
  cms.setHostValue(sliced, vSliced);
  cms.run(sg0);

  cms.getHostValue(dSliceable)
      .assertAllEquivalent(HostTensor::float32(0).expand(sliceable.shape()));

  cms.getHostValue(dSlice).assertAllEquivalent(vSliced.mul(2));
}
