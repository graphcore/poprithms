// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <gmock/gmock.h>

#include <testutil/common/compute/graph.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/simexecutable.hpp>

namespace {

using namespace poprithms::common::compute;

} // namespace

TEST(CommonComputeBasicSimExecutor, BasicReduceProduct) {
  using namespace poprithms::common::compute;
  test::Graph g;

  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.variable(DType::Int32, {2}, g.host());
  auto out = in0.reduceProduct(Shape({}));

  g.setRunnable({sg0});

  SimExecutable se(g);
  se.setHostValue<int>(in0, {2, 3});
  se.run(sg0);
  se.getHostValue(out).assertAllEquivalent(HostTensor::int32(6));
}

TEST(CommonComputeBasicSimExecutor, ViewChangeOps0) {
  test::Graph g;
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.variable(DType::Int32, {2, 3}, g.host());
  auto out0 = in0.slice_({0, 1}, {2, 3}).flatten_().add_(in0.constant(1.));

  g.setRunnable({sg0});
  SimExecutable se(g);

  auto v0 = HostTensor::int32({2, 3}, {0, 1, 2, 3, 4, 5});
  se.setHostValue(in0, v0.copy());
  se.run(sg0);
  se.getHostValue(out0).assertAllEquivalent(
      v0.slice_({0, 1}, {2, 3}).flatten_().add(1));
}

TEST(CommonComputeBasicSimExecutor, PadWithBroadcast0) {
  test::Graph g;
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.variable(DType::Int32, {2, 1}, g.host());
  auto out0 = in0.padWithBroadcastConstZero_({1, 0}, {0, 1});

  g.setRunnable({sg0});
  SimExecutable se(g);

  /**
   *
   *  [[10]
   *   [12]]
   *
   * to
   *
   *  [[ 0 0]
   *   [10 0]
   *   [12 0]]
   *
   * */
  auto v0 = HostTensor::int32({2, 1}, {10, 12});
  se.setHostValue(in0, v0.copy());
  se.run(sg0);
  se.getHostValue(out0).assertAllEquivalent(
      HostTensor::int32({3, 2}, {0, 0, 10, 0, 12, 0}));
}

TEST(CommonComputeBasicSimExecutor, MatMul0) {

  // The 2 tensors which we will multiply together with
  // (1) the compute::Graph (which we're testing).
  // (2) using the host tensor class (which we assume is correct).
  const auto t0 = HostTensor::uniformFloat64(-1, 1, {3, 4, 2}, 1011);
  const auto t1 = HostTensor::uniformFloat64(-1, 1, {2, 1, 2, 5}, 1011);

  // (2) the baseline:
  auto expected = t0.matmul(t1);

  // (1) Construct a computation graph with a matmul in it, construct a
  // SimExecutable and run it:
  test::Graph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0 = sg0.variable(DType::Float64, t0.shape(), g.host());
  auto in1 = sg0.variable(DType::Float64, t1.shape(), g.host());
  auto out = in0.matmul(in1);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(in0, t0);
  se.setHostValue(in1, t1);
  se.run(sg0);
  auto observed = se.getHostValue(out);

  observed.assertAllClose(expected, 1e-5, 1e-5);
}

TEST(CommonComputeBasicSimExecutor, MatMulDifferentOutType) {

  const auto t0 = HostTensor::randomInt32(-5, 5, {1, 2, 2, 4}, 1011);
  const auto t1 = HostTensor::randomInt32(-5, 5, {3, 1, 4, 3}, 1011);

  // (1) Construct a computation graph with a matmul in it, construct a
  // SimExecutable and run it:
  test::Graph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0 = sg0.variable(DType::Int32, t0.shape(), g.host());
  auto in1 = sg0.variable(DType::Int32, t1.shape(), g.host());
  auto out = in0.matmul(in1, DType::Int64, {});

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(in0, t0);
  se.setHostValue(in1, t1);
  se.run(sg0);
  auto observed = se.getHostValue(out);
  EXPECT_EQ(observed.dtype(), DType::Int64);
  EXPECT_EQ(observed.shape(), Shape({3, 2, 2, 3}));
}

TEST(CommonComputeBasicSimExecutor, RemainderIsFmod0) {

  const auto t0 = HostTensor::uniformFloat64(-5, 5, {20}, 1011);
  const auto t1 = HostTensor::uniformFloat64(-1, 1, {20}, 1012);

  test::Graph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0  = sg0.variable(DType::Float64, t0.shape(), g.host());
  auto in1  = sg0.variable(DType::Float64, t1.shape(), g.host());
  auto out0 = in0.rem(in1);
  auto out1 = in0.rem_(in1);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(in0, t0);
  se.setHostValue(in1, t1);

  se.run(sg0);
  auto observed0 = se.getHostValue(out0);
  auto observed1 = se.getHostValue(out1);

  auto t0_  = t0.getFloat64Vector();
  auto t1_  = t1.getFloat64Vector();
  auto out_ = observed0.getFloat64Vector();

  for (uint64_t i = 0; i < t0_.size(); ++i) {
    EXPECT_EQ(std::fmod(t0_.at(i), t1_.at(i)), out_.at(i));
  }

  observed0.assertAllEquivalent(t0.mod(t1));
  observed1.assertAllEquivalent(t0.mod(t1));
}
