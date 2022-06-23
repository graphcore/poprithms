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
using ::testing::AtLeast;

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
