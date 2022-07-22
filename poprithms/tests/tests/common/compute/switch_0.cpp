// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/testutil/finitedifference.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace {

using namespace poprithms::common::compute;

/**
 * Simple test case: 1 branch outputs sin(x) the other outputs cos(x).
 * */
void testSwitch0() {
  SlickGraph m;

  // callee 0: output sin of input
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto out0 = in0.sin();

  // callee 1: output cos of input
  auto sg1  = m.createSubGraph("sg1");
  auto in1  = sg1.hostFloat64Variable({});
  auto out1 = in1.cos();

  auto sg2      = m.createSubGraph("sg2");
  auto in2      = sg2.hostFloat64Variable({});
  auto cond     = sg2.hostInt32Variable({});
  auto swapOpId = sg2.switchOp(
      {sg0, sg1}, cond, {{in2, in0, 0}, {in2, in1, 1}}, {{out0, out1}});

  const auto out2 = out0.dstInCaller(CallEvent(swapOpId, sg0, 0));

  m.setRunnable({sg2});
  SimExecutable cms(m);

  auto ht = HostTensor::float64({}, {3.00});
  cms.setHostValue(in2, ht);

  cms.setHostValue<int32_t>(cond, {0});
  cms.run(sg2);
  cms.getHostValue(out2).assertAllClose(ht.sin(), 1e-7, 1e-7);

  cms.setHostValue(in2, ht);
  cms.setHostValue<int32_t>(cond, {1});
  cms.run(sg2);
  cms.getHostValue(out2).assertAllClose(ht.cos(), 1e-7, 1e-7);
}

/**
 * Basic test of training. Again the forward graph is:
 *
 *         +------------------------+
 *         | in0 ---> sin ---> out0 |
 * in2 ->  |                        | -> out2
 *         | in1 ---> cos ---> out1 |
 *         +------------------------+
 *
 * takes sin path if #cond is 0, else takes cos path.
 *
 * This test checks that dIn2 is cos (gradient of sin) if cond is 0,
 * and that it is -sin (gradient of cos) otherwise.
 * */
void testSwitchTrain0() {

  SlickGraph m;

  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto out0 = in0.sin();

  auto sg1  = m.createSubGraph("sg1");
  auto in1  = sg1.hostFloat64Variable({});
  auto out1 = in1.cos();

  auto sg2  = m.createSubGraph("sg2");
  auto in2  = sg2.hostFloat64Variable({});
  auto cond = sg2.hostInt32Variable({});
  auto swop =
      sg2.switchOp(/* callees */ {sg0, sg1},
                   /* condition */ cond,
                   /* input copies */ {{in2, in0, 0}, {in2, in1, 1}},
                   /* merged outputs */ {{in0, in1.id()}, {out0, out1}});

  const auto out2 = out0.dstInCaller(CallEvent(swop, sg0, 0));

  if (out2.id() != out1.dstInCaller(CallEvent(swop, sg1, 1))) {
    throw poprithms::test::error(
        "Destination of merged outputs should be same");
  }

  auto dIn2 = Autodiffer(m).backward(out2.id(), {in2.id()})[0];

  m.setRunnable({sg2});
  SimExecutable cms(m);

  auto ht = HostTensor::float64({}, {3.00});
  cms.setHostValue(in2, ht);
  cms.setHostValue<int32_t>(cond, {0});
  cms.run(sg2);
  cms.getHostValue(dIn2).assertAllClose(ht.cos(), 1e-7, 1e-7);

  cms.setHostValue<int32_t>(cond, {1});
  cms.run(sg2);
  cms.getHostValue(dIn2).assertAllClose(ht.sin().mul(-1), 1e-7, 1e-7);
}

void testTrainSwitchInCall0() {

  /**
   *
   * A switch within a call:
   *
   *  call(switch(sg0, sg1))
   *
   *  sg2 (the sub-graph with a switch in it:
   *
   *  . . . . . . . . . .
   *  . in2 --+----+    .
   *  . cond -+----+    .
   *  .       |    |    .
   *  .      sg0  sg1   .
   *  .       |    |    .
   *  .       +-+--+    .
   *  .         |       .
   *  .         v       .
   *  .       swapOut    .
   *  . . . . . . . . . .
   *
   *
   *  call3 = call(in3->in2, cond2->cond3)
   *
   * */

  SlickGraph m;

  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto out0 = in0.sin();

  auto sg1  = m.createSubGraph("sg1");
  auto in1  = sg1.hostFloat64Variable({});
  auto out1 = in1.cos();

  auto sg2       = m.createSubGraph("sg2");
  auto in2       = sg2.hostFloat64Variable({});
  auto cond2     = sg2.hostInt32Variable({});
  auto swapOpId2 = sg2.switchOp({sg0, sg1},
                                cond2,
                                {{in2, in0, 0}, {in2, in1, 1}},
                                {{in0, in1}, {out0, out1}});

  auto swapOut = out0.dstInCaller(CallEvent(swapOpId2, sg0, 0));

  auto sg3     = m.createSubGraph("sg3");
  auto cond3   = cond2.variable(sg3);
  auto in3     = sg3.hostFloat64Variable({});
  auto call3   = sg3.callAllOut(sg2, {{cond3, cond2}, {in3, in2}});
  auto callOut = swapOut.dstInCaller(call3);

  auto dIn3 = Autodiffer(m).backward(callOut.id(), {in3.id()})[0];

  m.setRunnable({sg3});
  SimExecutable cms(m);

  auto ht = HostTensor::float64({}, {3.00});
  cms.setHostValue(in3, ht);
  cms.setHostValue<int32_t>(cond3, {0});
  cms.run(sg3);
  cms.getHostValue(dIn3).assertAllClose(ht.cos(), 1e-7, 1e-7);
}

void testTrainAsymSwitch0() {

  SlickGraph m;
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({1, 1});
  auto out0 = in0.sin();

  auto sg1  = m.createSubGraph("sg1");
  auto in1  = in0.variable(sg1);
  auto in2  = in1.variable();
  auto out1 = in1.matmul(in2).relu();

  auto sg2    = m.createSubGraph("sg2");
  auto in3    = in1.variable(sg2);
  auto cond0  = sg2.hostInt32Variable({});
  auto swoony = sg2.switchOp({sg0, sg1},
                             cond0,
                             {{in3, in0, 0}, {in3, in1, 1}, {in3, in2, 1}},
                             {{out0, out1}},
                             {{{in0, 0}}, {{in1, 1}}, {{in2, 1}}});

  auto loss = out0.dstInCaller(CallEvent(swoony, sg0, 0)).reduceSum();
  auto dIn3 = Autodiffer(m).backward(loss, {in3})[0];

  m.setRunnable({sg2});
  SimExecutable cms(m);

  auto ht = HostTensor::float64({}, {3.00});
  cms.setHostValue(in3, ht);
  cms.setHostValue<int32_t>(cond0, {0});
  cms.run(sg2);
  cms.getHostValue(dIn3).assertAllClose(ht.cos(), 1e-7, 1e-7);

  // 2*ht for ht positive (simple maths).
  auto expected = ht.mul(2);
  cms.setHostValue(in3, ht);
  cms.setHostValue<int32_t>(cond0, {1});
  cms.run(sg2);
  cms.getHostValue(dIn3).assertAllClose(expected, 1e-7, 1e-7);
}

void testSwitchAllOut0() {
  SlickGraph m;

  auto sgFoo = m.createSubGraph("foo");
  (void)sgFoo;

  auto sg = m.createSubGraphs({"sg0", "sg1", "sg2"});

  auto in0 = sg[0].hostInt32Variable({});
  auto in1 = in0.variable(sg[1]);
  auto in2 = in0.variable(sg[2]);

  auto psi  = in1.pow(in1.constant(3));
  auto out1 = psi + in1.pow(in1.constant(2)).sin();

  auto out0 = in0 / (in0 + in0.constant(1) + in0.abs()).sqrt();

  auto cond = sg[2].hostInt32Variable({});
  auto sw   = sg[2].switchAllOut(
      {sg[1], sg[0]}, cond, {{in2, in1, 0}, {in2, in0, 1}}, {{out1, out0}});

  if (out1.dstInCaller(CallEvent(sw, sg[1], 0)).id() !=
      out0.dstInCaller(CallEvent(sw, sg[0], 1))) {
    throw poprithms::test::error(
        "Merged outputs have different ids - incorrect");
  }

  auto foo = psi.dstInCaller(CallEvent(sw, sg[1], 0));
  (void)foo;

  {
    bool caught{false};
    try {
      psi.dstInCaller(CallEvent(sw, sg[0], 1));
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      std::ostringstream oss;
      oss << "psi is not copied out of sg[0], failed to catch error";
      throw poprithms::test::error(oss.str());
    }
  }
}

} // namespace

int main() {
  testSwitch0();
  testTrainSwitchInCall0();
  testSwitchAllOut0();
  testTrainAsymSwitch0();
  testSwitchTrain0();
  return 0;
}
