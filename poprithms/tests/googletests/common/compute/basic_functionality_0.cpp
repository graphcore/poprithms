// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <gmock/gmock.h>

#include <testutil/common/compute/graph.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>

namespace {
using ::testing::AtLeast;

using namespace poprithms::common::compute;

TensorId
constant(Graph &g, SubGraphId sgId, DeviceId deviceId, const HostTensor &t) {
  TensorInfo info{t.shape(), deviceId, t.dtype()};
  const auto opId = g.nxtOpId();
  g.createComputeOp<ConstInit>({}, sgId, {info}, t);
  return {opId, 0};
}

TensorId
variable(Graph &g, SubGraphId sgId, DeviceId dId, const Shape &s, DType t) {
  TensorInfo info{s, dId, t};
  const auto opId = g.nxtOpId();
  g.createComputeOp<VarInit>({}, sgId, {info});
  return {opId, 0};
}

} // namespace

TEST(CommonComputeBasicFunctionality, ConstInit0) {
  using namespace poprithms::common::compute;
  test::Graph g;
  auto sgId = g.createSubGraphId("sg0");

  double v0{1.5};
  auto initVal = HostTensor::float64(v0);
  {

    auto const0 = constant(g, sgId, DeviceId(0), initVal);
    initVal.mul_(2);
    g.castOrThrow<ConstInit>(const0.opId())
        ->value()
        .assertAllEquivalent(HostTensor::float64(2 * v0));

    // Clone is not deep by default:
    auto const1 = g.clone(const0.opId(), {}, sgId);
    initVal.mul_(2);
    g.castOrThrow<ConstInit>(const1)->value().assertAllEquivalent(
        HostTensor::float64(4 * v0));

    {
      MemoryAliasMapper mam(g, {const0});
      const auto cols = mam.graph().colors(mam.id(const0));
      if (cols.size() != 1 || cols[0] != MemoryAliasConstant) {
        throw poprithms::test::error(
            "Constant should create constant in MemoryAliasMapper");
      }
    }
  }

  {
    // perform deep copy of value if you don't want changes in the HostTensor
    // to be reflected in the value stored by the ConstInit.
    auto const0 = constant(g, sgId, DeviceId(0), HostTensor::int32(1));
    initVal.add_(100);
    g.castOrThrow<ConstInit>(const0.opId())
        ->value()
        .assertAllEquivalent(HostTensor::int32(1));
  }
}

TEST(CommonComputeBasicFunctionality, ConstInit1) {

  using namespace poprithms::common::compute;

  auto getGraph = [](float v) {
    test::Graph g;
    auto sgId   = g.createSubGraphId("sg0");
    auto const0 = constant(g, sgId, DeviceId(0), HostTensor::float32(v));

    (void)const0;
    return g;
  };

  auto g0 = getGraph(1.223);
  auto g1 = getGraph(1.54);
  auto g2 = getGraph(1.54);
  EXPECT_NE(g0, g1);
  EXPECT_EQ(g1, g2);
}

TEST(CommonComputeBasicFunctionality, VarInit0) {
  test::Graph g;

  auto deviceId = g.host();
  auto sgId     = g.createSubGraphId("sg0");
  auto var0     = variable(g, sgId, deviceId, {3, 4}, DType::Float32);
  auto var1     = g.clone(var0.opId(), {}, sgId);

  std::cout << var0 << " " << var1 << std::endl;
  g.mutableCastOrThrow<VarInit>(var1)->setUserManagedHost(true);

  EXPECT_FALSE(g.castOrThrow<VarInit>(var0.opId())->isUserManagedHost());
  EXPECT_TRUE(g.castOrThrow<VarInit>(var1)->isUserManagedHost());

  auto ipuId = g.rootIpu();
  auto var2  = variable(g, sgId, ipuId, {3, 4}, DType::Float32);

  EXPECT_THROW(
      g.mutableCastOrThrow<VarInit>(var2.opId())->setUserManagedHost(true),
      poprithms::error::error);
}

TEST(CommonComputeBasicFunctionality, SubGraphTensor0) {

  test::Graph g;
  auto sg0 = g.createSubGraph("sg0");
  auto sg1 = g.createSubGraph("sg1");
  auto t0  = sg0.constant(HostTensor::int32(5), g.host());
  auto t1  = sg0.constant(DType::Int32, 5, g.host());
  (void)t1;
  auto t0_in_sg1 = t0.refTo_(sg1);
  EXPECT_TRUE(t0_in_sg1.graphIsSet());
  EXPECT_EQ(g.opIds<ConstInit>(sg0).size(), 2);
  EXPECT_EQ(g.opIds<RefFrom>(sg1).size(), 1);
}
