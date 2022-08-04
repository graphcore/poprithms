// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <gmock/gmock.h>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {

using namespace poprithms::common::compute;

} // namespace

// f(v)   = v^2.
// output = f(f(f(input))).
TEST(CommonComputeCall0, Test0) {

  SlickGraph g;

  // f(x) = v^2.
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.variable(DType::Int32, {2}, g.host());
  auto out0 = in0.pow(in0.constant(2));

  // Call f 3 times.
  auto sg1 = g.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);
  auto x   = in1;
  for (uint64_t i = 0; i < 3; ++i) {
    auto c0 = sg1.call(sg0, {{x, in0}}, {out0});
    x       = out0.dstInCaller(c0);
  }

  g.setRunnable({sg1});

  SimExecutable se(g);
  se.setHostValue<int>(in1, {2, 3});
  se.run(sg1);
  se.getHostValue(x).assertAllEquivalent(
      HostTensor::int32({2}, {2, 3}).pow(2).pow(2).pow(2));
}

TEST(CommonComputeCall0, TestPoplarStyleCall0) {

  SlickGraph g;

  // f(x) = v^2.
  auto sg0 = g.createSubGraph("sg0");
  auto t0  = sg0.variable(DType::Int32, {2}, g.host());
  t0.pow_(t0.constant(2));

  // A poplar style call without inputs or outputs.
  // It requires referencing a
  // single tensor in multiple sub-graphs.
  // Note that these often require special topological constraints, as there
  // are data (tensor->tensor) constraints to pin down the order of execution.
  auto sg1 = g.createSubGraph("sg1");
  auto t1  = t0.refTo_(sg1);
  for (uint64_t i = 0; i < 3; ++i) {
    sg1.call(sg0, {}, {});
  }

  g.setRunnable({sg1});

  SimExecutable se(g);
  se.setHostValue<int>(t1, {2, 3});
  se.run(sg1);
  se.getHostValue(t1).assertAllEquivalent(
      HostTensor::int32({2}, {2, 3}).pow(2).pow(2).pow(2));
}

TEST(CommonComputeCall0, BaseErrors0) {

  SlickGraph g;

  auto sg0        = g.createSubGraph("sg0");
  auto x0         = sg0.variable(DType::Int32, {}, g.host());
  auto x0float    = x0.variable(DType::Float32);
  auto x0big      = x0.variable(Shape({2, 3, 4}));
  auto x0constant = x0.constant(1.0);
  (void)x0constant;

  auto sg1 = g.createSubGraph("sg1");
  auto x1  = x0.variable(sg1);

  auto c0 = sg1.call(sg0, {{x1, x0}}, {x0});
  (void)c0;

  // Bad call, recursive:
  EXPECT_THROW(sg1.call(sg1, {{x1, x0}}, {x0}), poprithms::error::error);

  // Bad call, destination has different type:
  EXPECT_THROW(sg1.call(sg0, {{x1, x0float}}, {x0float}),
               poprithms::error::error);

  // Bad call, destination has different shape:
  EXPECT_THROW(sg1.call(sg0, {{x1, x0big}}, {x0big}),
               poprithms::error::error);

  // Bad call, destination is constant:
  // TODO(jn): not currently tested for.
  /*
  EXPECT_THROW(sg1.call(sg0, {{x1, x0constant}}, {x0constant}),
               poprithms::error::error);
  */

  // Bad copy in destination:
  EXPECT_THROW(sg1.call(sg0, {{x1, x1}}, {x0}), poprithms::error::error);

  // Bad copy in source:
  EXPECT_THROW(sg1.call(sg0, {{x0, x0}}, {x0}), poprithms::error::error);

  // Bad copy out source:
  EXPECT_THROW(sg1.call(sg0, {{x0, x1}}, {x1}), poprithms::error::error);
}

TEST(CommonComputeCall0, RepeatCopiesRegistered0) {

  SlickGraph g;
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.hostInt32Variable({});
  auto out0 = in0.cos().sin();

  auto sg1 = g.createSubGraph("sg1");
  auto in1 = sg1.hostInt32Variable({10});
  auto rpt =
      sg1.repeat(sg0, 10, {{in1, in0}}, {}, {{out0, IsStackedCopy::Yes}});

  auto out1 = out0.dstInCaller(rpt);
  (void)out1;

  CallEvent ce(rpt, sg0, CalleeIndex(0));
  const auto copyOuts = g.computeOp(out0.opId()).outCopies(out0.outIndex());
  EXPECT_EQ(copyOuts.size(), 1);
  EXPECT_EQ(copyOuts.back(), ce);
  g.verifyValid();
}
