// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <gmock/gmock.h>
#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/scheduler.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {

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
  SlickGraph g;
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
    SlickGraph g;
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
  SlickGraph g;

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

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto sg1 = g.createSubGraph("sg1");
  auto t0  = sg0.constant(HostTensor::int32(5), g.host());
  auto t1  = sg0.constant(DType::Int32, 5, g.host());
  (void)t1;
  auto t0_in_sg1 = t0.refTo_(sg1);
  EXPECT_TRUE(t0_in_sg1.graphIsSet());
  EXPECT_EQ(g.opIds<ConstInit>(sg0).size(), 2);
  EXPECT_EQ(g.opIds<RefFrom>(sg1).size(), 1);

  auto t2 = t0.reshape_({1, 1, 1});
  EXPECT_EQ(t2.shape().rank_i64(), 3ll);
  EXPECT_THROW(t0.reshape_({4, 5, 6}), poprithms::error::error);
}

TEST(CommonComputeBasicFunctionality, InsertViewChangeIdentity) {

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto t0  = sg0.constant(
      HostTensor::uniformFloat32(-1, 1, {2, 1, 3, 1, 4}, 1011), g.host());

  // ConstInit.
  EXPECT_EQ(g.nOps(), 1);
  auto t1 = t0.dimShuffle_({{0, 2, 4, 1, 3}});

  // DimShuffle.
  EXPECT_EQ(g.nOps(), 2);

  // Identities, so expect not to have any new ops added.
  t1.dimShuffle_({{0, 1, 2, 3, 4}});
  EXPECT_EQ(g.nOps(), 2);

  auto t3 = t0.dimShuffle_({{0, 3, 2, 1, 4}});
  EXPECT_EQ(g.nOps(), 2);
  EXPECT_EQ(t3.id(), t0.id());

  // Identity reversals:
  t0.reverse_(3);
  EXPECT_EQ(g.nOps(), 2);
  t0.reverse_(Dimensions({0, 1, 0}));
  EXPECT_EQ(g.nOps(), 2);

  // A non-identity reversal:
  t0.reverse_(2);
  EXPECT_EQ(g.nOps(), 3);

  // Identity reshape:
  t0.reshape_(t0.shape());
  EXPECT_EQ(g.nOps(), 3);

  // Identity slice:
  t0.slice_({0, 0, 0, 0, 0}, {2, 1, 3, 1, 4});
  EXPECT_EQ(g.nOps(), 3);

  // Check that dimensions are canonicalized correctly. Even number of 4's,
  // odd number of 2's.
  auto foo = t0.reverse_(Dimensions({2, 4, 2, 2, 4}));
  EXPECT_EQ(g.dynamicCast<Reverse_>(foo.id().opId())->dimensions(),
            Dimensions({2}));
}

TEST(CommonComputeBasicFunctionality, Scheduler) {

  // Test of circular referencing.
  {
    SlickGraph g;
    auto sg0 = g.createSubGraph("sg0");
    auto sg1 = g.createSubGraph("sg1");

    auto t0 = sg0.constant(HostTensor::uniformFloat32(-1, 1, {2, 3}, 1011),
                           g.host());
    auto t1 = t0.refTo_(sg1);
    auto t2 = t1.reduceMin();
    t2.refTo_(sg0);

    // there is a cycle in the graph
    EXPECT_THROW(Scheduler::scheduleByRefs(g), poprithms::error::error);
  }

  // Test of graph with no compute.
  {
    SlickGraph g;
    auto sg0 = g.createSubGraph("sg0");
    auto t0  = sg0.constant(DType::Float32, 1., g.host());
    t0.reshape_({1, 1, 1}).expand_({1, 2, 3, 4, 5});
    EXPECT_EQ(Scheduler::vanillaComputeSchedule(g, sg0).size(), 0);
  }
}

TEST(CommonComputeBasicFunctionality, BadConcats0) {
  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto v0  = sg0.variable(DType::Int32, {3, 4}, g.host());
  auto v1  = sg0.variable(DType::Int32, {4, 5}, g.host());
  EXPECT_THROW(Tensor::concat_({v0, v1}, 0), poprithms::error::error);
  EXPECT_THROW(Tensor::concat_({v0, v0}, 2), poprithms::error::error);
  EXPECT_THROW(Tensor::concat_({}, 0), poprithms::error::error);
}

TEST(CommonComputeBasicFunctionality, RefFrom0) {
  SlickGraph m;

  auto sg0 = m.createSubGraph("sg0");
  auto sg1 = m.createSubGraph("sg1");
  auto sg2 = m.createSubGraph("sg2");

  const auto in0 = sg0.hostFloat32Variable({}).relu();
  const auto in1 = sg1.hostFloat32Variable({}).sqrt();
  const auto in2 = sg2.hostFloat32Variable({}).abs();

  const auto ref0to1 = in0.refTo_(sg1.id());
  const auto ref1to2 = in1.refTo_(sg2.id());
  const auto ref2to0 = in2.refTo_(sg0.id());
  (void)ref1to2;

  EXPECT_FALSE(ref0to1.hasDerivedRefs());
  EXPECT_FALSE(ref0to1.isRootRef());
  EXPECT_TRUE(in0.hasDerivedRefs());

  const auto base = sg0.hostFloat32Variable({});
  EXPECT_TRUE(base.isRootRef());
  EXPECT_TRUE(base.refsExcludingSelf().empty());
  EXPECT_EQ(ref2to0.refsExcludingSelf().size(), 1);
  EXPECT_EQ(in2.refsExcludingSelf().size(), 1);

  // one variable, referenced in 3 graphs.
  SlickGraph m2;
  auto sgs = m2.createSubGraphs({"sg0", "sg1", "sg2", "sg3"});
  auto v   = sgs[0].hostFloat32Variable({});
  Tensors r0s;
  for (auto sg : sgs) {
    r0s.push_back(v.refTo_(sg));
  }
  Tensors r1s;
  for (auto r : r0s) {
    for (auto sg : sgs) {
      r1s.push_back(r.refTo_(sg));
    }
  }

  EXPECT_EQ(m2.opIds<RefFrom>().size(), 3);
  EXPECT_NE(m, m2);
}

TEST(CommonComputeBasicFunctionality, RemoteTensorErrors0) {

  SlickGraph g;
  auto sg0      = g.createSubGraph("sg0");
  auto ipu0     = sg0.rootIpuFloat32Variable({6, 3});
  auto indices0 = sg0.variable(DType::Unsigned32, {2}, g.rootIpu());
  auto indices1 = sg0.variable(DType::Unsigned32, {6}, g.rootIpu());

  auto rem = ipu0.ipuToRemote(indices1, 2, {});
  (void)rem;

  EXPECT_EQ(rem.shape(), Shape({2, 3}));
  EXPECT_THROW(ipu0.ipuToRemote(indices0, 4, {}), poprithms::error::error);

  EXPECT_THROW(ipu0.reshape({2, 3, 3}).ipuToRemote(indices0, 2, {}),
               poprithms::error::error);

  EXPECT_THROW(rem + rem, poprithms::error::error);
}
