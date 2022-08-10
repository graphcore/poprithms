// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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
} // namespace

int main() {
  testTraversal0();
  std::make_unique<SimTester<RepeatTester>>()->all();
}
