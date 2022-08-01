// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/testutil/finitedifference.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace {

using namespace poprithms::common::compute;

void expect_true(bool c, const std::string &s0) {
  if (!c) {
    throw poprithms::test::error("Failed in expect_true with: " + s0);
  }
}

void test0() {
  SlickGraph g;
  auto sg = g.createSubGraph("sg");

  auto in0 = sg.hostFloat32Variable({});
  auto b   = in0.relu();
  auto c   = in0.zero_();
  auto d   = b + c;

  expect_true(g.isValueDependent({0, d.opId(), 0}),
              "traversal of add is value dependent");

  expect_true(!g.isValueDependent({0, c.opId(), 0}),
              "traversal of zero_ is not value dependent");
}

void testCall0() {

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.hostInt32Variable({});
  auto in1 = in0.variable();
  auto in2 = in0.variable();

  // value of in0 doesn't change out0.
  auto out0 = in0.update_(in2).pow(2);

  // value of in1 does change out1.
  auto out1 = in1.relu();

  auto sg1  = g.createSubGraph("sg1");
  auto in0_ = in0.variable(sg1);
  auto in1_ = in0_.variable();
  auto c0   = sg1.call(sg0, {{in0_, in0}, {in1_, in1}}, {out0, out1});

  expect_true(!g.isValueDependent({0, c0, 0}), "input 0 is written to");
  expect_true(g.isValueDependent({1, c0, 1}), "input 1 is used in output 0");
}

void testRepeat0() {

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0 = sg0.hostInt32Variable({});
  auto rel = in0.relu();
  auto zer = rel.zero_();

  auto sg1 = g.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);

  //
  // in0 ---->  x  -----> zero_.
  //  ^                     v
  //  |                     |
  //  +---------------------+
  //

  for (auto isc : {IsStackedCopy::Yes, IsStackedCopy::No}) {

    auto rpt = sg1.repeat(sg0,
                          10,
                          {},
                          {{{in1.id(), in0.id(), zer.id()}}},
                          {{in0, isc}, {rel, isc}, {zer.id(), isc}});

    if (isc == IsStackedCopy::Yes) {
      expect_true(g.isValueDependent({0, rpt, 0}),
                  "the input copied straight out");
      expect_true(g.isValueDependent({0, rpt, 1}), "relu of the input");
      expect_true(!g.isValueDependent({0, rpt, 2}),
                  "the output of the zero inplace");
    } else {
      expect_true(
          g.isValueDependent({0, rpt, 1}),
          "final value of relu of the input. Actually we expect this to NOT "
          "be dependent on the input as a zero value is carried back. But "
          "the implementation we use doesn't check this currently "
          "(conservative).");

      expect_true(!g.isValueDependent({0, rpt, 2}),
                  "the output of the zero inplace (non-stacked)");
    }
  }
}

void testTrain0() {

  SlickGraph g;
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.hostFloat32Variable({});
  auto out0 = (in0.relu().add(1) + in0.zero_().add(2));

  bool caught{false};
  try {
    Autodiffer ad(g);
    ad.backwardInGraph({out0}, {}, {in0}, {in0.constant(1)});
  } catch (const poprithms::error::error &e) {
    caught        = true;
    std::string w = e.what();
    if (w.find("insufficient checkpointing") == std::string::npos) {
      throw poprithms::test::error(
          "Expected the message to say something about why autodiff failed - "
          "not enough checkpoints");
    }
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error about insufficient checkpointing");
  }
}

// OpId  Name                OpType                 InTensors    Shape
// ----  ----                ------                 ---------    -----
// 0                         VarInit                ()           ()
// 1                         VarInit                ()           ()
// 2                         VarInit                ()           (4,4)
// 3                         Mul                    ops=(1,0)    ()
// 4                         Expand_                ops=(3)      (4,4)
// 5                         CopyFrom_              ops=(2,4)    (4,4)
// 6                         Sin                    ops=(5)      (4,4)
// 7                         ReduceSum(dims=(0,1))  ops=(6)      (1,1)
// 8                         Reshape_               ops=(7)      ()
// 9                         ConstInit(1.000000)    ()           ()
// 10  rerun/2               VarInit                ()           (4,4)
// 11  rerun/3               Mul                    ops=(1,0)    ()
// 12  rerun/4               Expand_                ops=(11)     (4,4)
// 13  rerun/5               CopyFrom_              ops=(10,12)  (4,4)
// 14  grad-of-op-8-input-0  Reshape_               ops=(9)      (1,1)
// 15  grad-of-op-7-input-0  Expand_                ops=(14)     (4,4)
// 16                        Cos                    ops=(13)     (4,4)
// 17  grad-of-op-6-input-0  Mul                    ops=(16,15)  (4,4)
// 18  grad-of-op-5-input-1  ReduceSum(dims=())     ops=(17)     (4,4)
// 19                        ReduceSum(dims=(0,1))  ops=(18)     (1,1)
// 20  grad-of-op-4-input-0  Reshape_               ops=(19)     ()
// 21                        Mul                    ops=(20,0)   ()
// 22  grad-of-op-3-input-0  ReduceSum(dims=())     ops=(21)     ()
// 23                        Mul                    ops=(20,1)   ()
// 24  grad-of-op-3-input-1  ReduceSum(dims=())     ops=(23)     ()

void testTrain1() {

  // No checkpoints. This is fine because the input is written to. We confirm
  // this.
  SlickGraph g;
  auto sg0  = g.createSubGraph("sg0");
  auto w0   = sg0.hostFloat32Variable({});
  auto in0  = w0.variable();
  auto loss = w0.variable({4, 4})
                  .update_(in0.mul(w0).expand_({4, 4}))
                  .sin()
                  .reduceSum(Shape{});
  Autodiffer(g).backwardInGraph({loss}, {w0, in0}, {w0}, {in0.constant(1)});

  if (g.opIds<VarInit>().size() != 4) {
    std::ostringstream oss;
    sg0.append(oss);
    oss << "\n\nExpected exactly 4 VarInits, 3 from the original graph and "
           "one recomputed";
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {
  test0();
  testCall0();
  testTrain0();
  testTrain1();
  testRepeat0();
}
