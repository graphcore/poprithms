// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/callstackquerier.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/testutil/repeattester.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/stacktensorid.hpp>
#include <poprithms/program/callstack/stackutil.hpp>

namespace {

using namespace poprithms::common::compute::testutil;
using namespace poprithms::common::compute;

void testTraversal0() {

  using poprithms::program::callstack::StackTensorId;

  SlickGraph g;
  auto sg0 = g.createSubGraph("caller");
  auto sg1 = g.createSubGraph("callee");

  auto in10 = sg1.hostFloat32Variable({});
  auto a    = in10.pow(2);
  auto b    = a.pow(2);
  auto c    = b.pow(2);

  auto in00 = sg0.hostFloat32Variable({});

  auto rpt = sg0.repeat(
      sg1, 10, {}, {{{in00.id(), in10, c}}}, {{c, IsStackedCopy::Yes}});
  auto out = c.dstInCaller(rpt);

  using poprithms::program::callstack::StackUtil;

  CallstackQuerier q(g);

  {
    auto obs = StackUtil::tensorIds(
        q.onMultiGraphPathFrom(StackUtil::inMainScope({in00})));
    if (obs.count(out) != 1) {
      throw poprithms::test::error("Failed to traverse to output from input");
    }
  }
  {

    auto obs = StackUtil::tensorIds(
        q.onMultiGraphPathFrom({StackTensorId(c, {CallEvent(rpt, sg0, 0)})}));
    if (obs.count(in10) != 1) {
      throw poprithms::test::error("Failed to traverse back to the repeat "
                                   "input through the carry edge");
    }
  }
}

// test with a carried input, where there is no compute in the callee
void testNoComputeInRepeat0() {
  SlickGraph g;
  auto caller  = g.createSubGraph("caller");
  auto xCaller = caller.hostFloat32Variable({});
  auto callee  = g.createSubGraph("callee");
  auto xCallee = callee.hostFloat32Variable({});

  auto rptOp = caller.repeat(callee,
                             2,
                             {},
                             {{{xCaller, xCallee, xCallee.id()}}},
                             {{xCallee, IsStackedCopy::No}});

  auto loss = (xCallee.dstInCaller(rptOp));

  Autodiffer ad(g);
  auto dx = ad.backward(loss, {xCaller})[0];
  g.setRunnable({caller});

  SimExecutable se(g);
  se.setHostValue(xCaller, HostTensor::float32({}, {7.}));
  se.run(caller);
  se.getHostValue(dx).assertAllEquivalent(HostTensor::float32({}, {1.}));
}

// test with a stacked input, where there is no compute in the callee
void testNoComputeInRepeat1() {
  SlickGraph g;
  auto caller  = g.createSubGraph("caller");
  auto xCaller = caller.hostFloat32Variable({4});
  auto callee  = g.createSubGraph("callee");
  auto xCallee = callee.hostFloat32Variable({});

  auto rptOp = caller.repeat(
      callee, 4, {{xCaller, xCallee}}, {}, {{xCallee, IsStackedCopy::Yes}});

  auto loss = (xCallee.dstInCaller(rptOp).reduceSum(Shape{}));

  Autodiffer ad(g);
  auto dx = ad.backward(loss, {xCaller})[0];
  g.setRunnable({caller});

  SimExecutable se(g);
  se.setHostValue(xCaller, HostTensor::float32({4}, {7., 6, 5, 4.}));
  se.run(caller);
  se.getHostValue(dx).assertAllEquivalent(HostTensor::float32(1).expand({4}));
}

} // namespace

int main() {
  testTraversal0();
  testNoComputeInRepeat0();
  testNoComputeInRepeat1();
  std::make_unique<SimTester<RepeatTester>>()->all();
}
