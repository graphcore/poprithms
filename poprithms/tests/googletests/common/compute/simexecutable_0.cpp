// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
