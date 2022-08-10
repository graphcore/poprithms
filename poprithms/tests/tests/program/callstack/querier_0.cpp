// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>

#include <testutil/program/callstack/graph.hpp>
#include <testutil/program/callstack/querier.hpp>

#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/callstack/querier.hpp>
#include <poprithms/program/callstack/stackutil.hpp>

namespace {

using namespace poprithms::program;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::program::callstack::CalleeIndex;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CopyIn;
using poprithms::program::callstack::CopyIns;
using poprithms::program::callstack::CopyOuts;
using poprithms::program::callstack::StackTensorId;
using poprithms::program::callstack::StackTensorIds;

void testOnMultiGraphPathTo0() {

  /**
   *
   * sg0 : in0 -> out0.
   *
   * sg1 : in1 -> sg0 -> out1.
   *
   * sg2 : in2 -> sg1 -> out2.
   *
   * */

  callstack_test::Graph m;
  auto sg0        = m.createSubGraphId("g0");
  const auto in0  = m.insert({}, 1, sg0, "in0");
  const auto out0 = TensorId(m.insert({{in0, 0}}, 1, sg0, "out0"), 0);

  auto sg1       = m.createSubGraphId("g1");
  const auto in1 = m.insert({}, 1, sg1, "in1");
  const TensorId out1{m.insert(sg1,
                               {sg0},
                               CopyIns{{{{in1, 0}, {in0, 0}, 0}}},
                               CopyOuts{{{out0}}},
                               OptionalTensorId{},
                               {},
                               "out1"),
                      0};

  auto sg2       = m.createSubGraphId("g2");
  const auto in2 = m.insert({}, 1, sg2, "in2");
  const TensorId out2{m.insert(sg2,
                               {sg1},
                               CopyIns{{{{in2, 0}, {in1, 0}, 0}}},
                               CopyOuts{{{out1}}},
                               {},
                               {},
                               "out2"),
                      0};

  auto stacky = callstack_test::Querier(m).onMultiGraphPathTo(
      callstack::StackUtil::inMainScope({out2}));

  CallEvent out2_{out2.opId(), sg1, 0};
  CallEvent out1_{out1.opId(), sg0, 0};

  StackTensorIds expected{
      StackTensorId{out2, {}},      // <- target of "onMultiGraphPathTo"
      StackTensorId{out1, {out2_}}, // <- source of above  (copy out)
      StackTensorId{out0, {out2_, out1_}}, // <- source of above (copy out)
      StackTensorId{{in0, 0}, {out2_, out1_}}, // <- source of above (in sg0)
      StackTensorId{{in1, 0}, {out2_}},        // <- source of above (copy in)
      StackTensorId{{in2, 0}, {}}}; // <- source of above (copy in).

  std::sort(stacky.begin(), stacky.end());
  std::sort(expected.begin(), expected.end());

  if (stacky != expected) {
    std::ostringstream oss;
    oss << "Expected\n" << expected << " but observed\n" << stacky << ". ";
    throw poprithms::test::error(oss.str());
  }

  auto stackBack = callstack_test::Querier(m).onMultiGraphPathFrom(
      callstack::StackUtil::inMainScope({{in2, 0}}));

  std::sort(stackBack.begin(), stackBack.end());

  if (stackBack != expected) {
    std::ostringstream oss;
    oss << "For the forwards traversal, expected \n"
        << expected << " but observed\n"
        << stackBack << ". ";
    throw poprithms::test::error(oss.str());
  }
}

void testNoPathToTarget0() {

  callstack_test::Graph m;

  //
  //
  //          sg0
  //  +.................+
  //  .                 .
  //  .  in0        in1 .    +----in2  in3
  //  .   |          |  .    |      |  |
  //  .   +-+-----+--+  .    v      call(sg0) ->-+
  //  .     |     |     .    |      |            |
  //  .   add    sub    .    | call0,0        call0,1
  //  .                 .    |     |
  //  +.................+    |     v
  //                         +--> call(sg0) ---> (call1,0, call1,1).
  //
  //

  const auto sg0 = m.createSubGraphId("g0");
  const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};
  const TensorId in1{m.insert({}, 1, sg0, "in1"), 0};
  const TensorId add{m.insert({in0, in1}, 1, sg0, "add"), 0};
  const TensorId sub{m.insert({in0, in1}, 1, sg0, "sub"), 0};

  const auto sg1 = m.createSubGraphId("g1");
  const TensorId in2{m.insert({}, 1, sg1, "in2"), 0};
  const TensorId in3{m.insert({}, 1, sg1, "in3"), 0};

  auto call0 = m.insert(sg1,   // <-- calling graph
                        {sg0}, // <-- callee graph(s)
                        CopyIns{{{in2, in0, CalleeIndex(0)}, {in3, in1, 0}}},
                        CopyOuts{{{add}, {sub}}},
                        {},
                        {},
                        "call0");

  auto call1 = m.insert(sg1,
                        {sg0},
                        CopyIns{{{{call0, 0}, in0, 0}, {in2, in1, 0}}},
                        CopyOuts{{{add}, {sub}}},
                        {},
                        {},
                        "call1");

  // expect every tensor except call0, 1
  {
    const auto stacky = callstack_test::Querier(m).onMultiGraphPathTo(
        callstack::StackUtil::inMainScope(m.outTensorIds(call1)));
    auto counts = callstack::StackUtil::getCounts(stacky);
    if (counts.size() != 9) {
      throw poprithms::test::error(
          "Expected 3/10 tensors to be visited, not " +
          std::to_string(counts.size()));
    }
  }

  {
    const auto stacky = callstack_test::Querier(m).onMultiGraphPathFrom(
        callstack::StackUtil::inMainScope({in2}));

    auto counts = callstack::StackUtil::getCounts(stacky);
    if (counts.size() != 9) {
      throw poprithms::test::error(
          "Expected 9/10 tensors to be visited, all except in3. Not " +
          std::to_string(counts.size()));
    }
  }

  // Check that custom #accept works:
  {
    const auto stacky = callstack_test::Querier(m).onMultiGraphPathFrom(
        callstack::StackUtil::inMainScope({in2}),
        [&](auto x) { return x.tId() != add && x.tId() != sub; });

    auto counts = callstack::StackUtil::getCounts(stacky);
    if (counts.size() != 3) {
      throw poprithms::test::error(
          "Expected 3/10 tensors to be visited: in2, in0 and in1. Not " +
          std::to_string(counts.size()));
    }
  }
}

void testNestedFullStack0() {
  callstack_test::Graph m;
  const auto sg0 = m.createSubGraphId("sg0");
  const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};

  const auto sg1 = m.createSubGraphId("sg1");
  const TensorId in1{m.insert({}, 1, sg1, "in1"), 0};

  m.insert(sg1, {sg0}, CopyIns{}, CopyOuts{}, {}, {}, "call0");

  if (callstack_test::Querier(m)
          .onMultiGraphPathTo(callstack::StackUtil::inMainScope({in1}))
          .size() != 1) {
    throw poprithms::test::error("in1 should not be reached on multi-graph "
                                 "path as sg0 has no inputs or outputs");
  }

  if (callstack_test::Querier(m).nestedFullStack({sg1}).size() != 2) {
    throw poprithms::test::error(
        "in0 should be reached with nestedFullStack");
  }
}

void assertOnMultiGraph(const StackTensorIds &observed,
                        const StackTensorIds &expected) {
  std::set<StackTensorId> sObserved(observed.cbegin(), observed.cend());
  std::set<StackTensorId> sExpected(expected.cbegin(), expected.cend());
  if (sObserved != sExpected) {
    std::ostringstream oss;
    oss << "Expected the StackTensorIds on the multi-graph path to be "
        << expected << " but observed " << observed << ".";
    throw poprithms::test::error(oss.str());
  }
}

void testMultiGraphBack0() {

  callstack_test::Graph m;

  // in0 ---> out0
  // in1 ---> out1
  const auto sg0 = m.createSubGraphId("sg0");
  const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};
  const TensorId in1{m.insert({}, 1, sg0, "in1"), 0};
  const TensorId out0{m.insert({in0}, 1, sg0, "out0"), 0};
  const TensorId out1{m.insert({in1}, 1, sg0, "out1"), 0};

  // in2 ---+--> out2
  //        |
  //        +--> out3
  const auto sg1 = m.createSubGraphId("sg1");
  const TensorId in2{m.insert({}, 1, sg1, "in2"), 0};
  auto out23 = m.insert({in2}, 2, sg1, "out2,3");
  TensorId out2{out23, 0};
  TensorId out3{out23, 1};

  const auto sg2 = m.createSubGraphId("sg2");
  const TensorId in3{m.insert({}, 1, sg2, "in3"), 0};
  const TensorId in4{m.insert({}, 1, sg2, "in4"), 0};

  // case 0: copy in3->in0, in4->1, return out0, out1
  // case 1: copy in3->in2          return out2, out3
  // case 2: copy in4->in2          return out3, out2.
  const auto sw = m.insert(
      sg2,
      {sg0, sg1, sg1},
      CopyIns(
          //          case0          case0          case1          case2
          {CopyIn(in3, in0, 0), {in4, in1, 0}, {in3, in2, 1}, {in4, in2, 2}}),
      //                             case0 case1 case2
      //                               |     |     |
      CopyOuts(std::vector{TensorIds{out0, out2, out3},   // <- OutIndex(0)
                           TensorIds{out1, out3, out2}}), // <- OutIndex(1)
      {},
      {},
      "switchWith3");

  {
    // Paths to output 0 of the switch. How is this calculated? Just work back
    // through each of the switch cases starting from {sw,0}.
    StackTensorIds expected{StackTensorId({sw, 0}, {}),
                            // case0:
                            {out0, {CallEvent(sw, sg0, 0)}},
                            {in0, {CallEvent(sw, sg0, 0)}},
                            {in3, {}},

                            // case1:
                            {out2, {CallEvent(sw, sg1, 1)}},
                            {in2, {CallEvent(sw, sg1, 1)}},
                            {in3, {}},

                            // case2:
                            {out3, {CallEvent(sw, sg1, 2)}},
                            {in2, {CallEvent(sw, sg1, 2)}},
                            {in4, {}}};

    auto obs = callstack_test::Querier(m).onMultiGraphPathTo(
        callstack::StackUtil::inMainScope({{sw, 0}}));

    assertOnMultiGraph(obs, expected);
  }

  {
    // Paths to output 1 of the switch.
    StackTensorIds expected{StackTensorId({sw, 1}, {}),
                            // case0:
                            {out1, {CallEvent(sw, sg0, 0)}},
                            {in1, {CallEvent(sw, sg0, 0)}},
                            {in4, {}},

                            {out3, {CallEvent(sw, sg1, 1)}},
                            {in2, {CallEvent(sw, sg1, 1)}},
                            {in3, {}},

                            {out2, {CallEvent(sw, sg1, 2)}},
                            {in2, {CallEvent(sw, sg1, 2)}},
                            {in4, {}}

    };

    auto obs = callstack_test::Querier(m).onMultiGraphPathTo(
        callstack::StackUtil::inMainScope({{sw, 1}}));

    assertOnMultiGraph(obs, expected);
  }
}

void testMultiGraphBack1() {

  callstack_test::Graph m;

  // in0 ---> out0
  // in1 ---> out1
  const auto sg0 = m.createSubGraphId("sg0");
  const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};
  const TensorId in1{m.insert({}, 1, sg0, "in1"), 0};
  const TensorId out0{m.insert({in0}, 1, sg0, "out0"), 0};
  const TensorId out1{m.insert({in1}, 1, sg0, "out1"), 0};

  const auto sg1 = m.createSubGraphId("sg1");
  const TensorId in2{m.insert({}, 1, sg1, "in2"), 0};

  const auto c0 =
      m.insert(sg1, {sg0}, {{{in2, in0, 0}}}, {{{out0}}}, {}, {}, "call0");

  const auto c1 = m.insert(sg1,
                           {sg0},
                           CopyIns({CopyIn({c0, 0}, in1, 0)}),
                           CopyOuts(std::vector<TensorIds>{{out1}}),
                           {},
                           {},
                           "call0");

  auto obs = callstack_test::Querier(m).onMultiGraphPathTo(
      callstack::StackUtil::inMainScope({{c1, 0}}));

  StackTensorIds expected{StackTensorId{{c1, 0}, {}},
                          StackTensorId{out1, {CallEvent(c1, sg0, 0)}},
                          StackTensorId{in1, {CallEvent(c1, sg0, 0)}},
                          StackTensorId{{c0, 0}, {}},
                          StackTensorId{out0, {CallEvent(c0, sg0, 0)}},
                          StackTensorId{in0, {CallEvent(c0, sg0, 0)}},
                          StackTensorId{in2, {}}};

  assertOnMultiGraph(obs, expected);
}

void testSwitch0() {

  /**
   *
   * in1  ---+-- in0 ---> copy out.
   *         |
   * cond ---+
   *
   *
   * Check that "cond" is found in the backwards search.
   * */
  callstack_test::Graph m;
  const auto sg0 = m.createSubGraphId("sg0");
  const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};

  const auto sg1 = m.createSubGraphId("sg1");
  const TensorId in1{m.insert({}, 1, sg1, "in1"), 0};
  const TensorId cond{m.insert({}, 1, sg1, "cond"), 0};
  const auto c0 =
      m.insert(sg1, {sg0}, {{{in1, in0, 0}}}, {{{in0}}}, cond, {}, "switch");

  auto obs = callstack_test::Querier(m).onMultiGraphPathTo(
      callstack::StackUtil::inMainScope({{c0, 0}}));

  StackTensorIds expected{
      StackTensorId{in0, {CallEvent(c0, sg0, 0)}},
      StackTensorId{in1, {}},
      StackTensorId{cond, {}},
      StackTensorId{{c0, 0}, {}},
  };
  assertOnMultiGraph(obs, expected);
}

void testRepeat0() {

  auto test = [](bool carryFromEnd) {
    callstack_test::Graph m;
    const auto sg0 = m.createSubGraphId("sg0");

    //
    //   in1
    //    |
    //    |
    //    v
    // copied in
    //    |
    //    v
    //   in0 --> x1 --> x2
    //           |
    //           |
    //           v
    //        copied out
    //
    //
    // if carry x2 -> in0 : x2 should be visited.
    // if carry x1 -> in0 : x2 should not be visited.
    //

    const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};
    const TensorId x1{m.insert({in0}, 1, sg0, "x1"), 0};
    const TensorId x2{m.insert({x1}, 1, sg0, "x2"), 0};

    const auto sg1 = m.createSubGraphId("sg1");
    const TensorId in1{m.insert({}, 1, sg1, "in1"), 0};

    auto carrySource = carryFromEnd ? x2 : x1;
    const auto c0    = m.insert(sg1,
                             {sg0},
                             {{{in1, in0, 0}}},
                             {{{x1}}},
                             {},
                             {{carrySource, in0}},
                             "repeat");

    auto obs = callstack_test::Querier(m).onMultiGraphPathTo(
        callstack::StackUtil::inMainScope({{c0, 0}}));

    if (carrySource == x2) {
      if (std::find(obs.cbegin(),
                    obs.cend(),
                    StackTensorId(x2, {CallEvent(c0, sg0, 0)})) ==
          obs.cend()) {
        throw poprithms::test::error(
            "The carry source (x2) should be visited");
      }
    }

    else {

      if (std::find(obs.cbegin(),
                    obs.cend(),
                    StackTensorId(x2, {CallEvent(c0, sg0, 0)})) !=
          obs.cend()) {
        throw poprithms::test::error(
            "x2 is not on a path to to target repeat output");
      }
    }
  };
  test(true);
  test(false);
}

void testScheduling0() {

  callstack_test::Graph m;
  const auto sg0 = m.createSubGraphId("sg0");
  const TensorId in0{m.insert({}, 1, sg0, "in0"), 0};
  const TensorId x01{m.insert({in0}, 1, sg0, "x1"), 0};
  const TensorId x02{m.insert({in0}, 1, sg0, "x2"), 0};
  const TensorId x03{m.insert({in0}, 1, sg0, "x3"), 0};
  const TensorId out0{m.insert({x01, x02, x03}, 1, sg0, "out0"), 1};

  const auto sg1 = m.createSubGraphId("sg1");
  const TensorId in1{m.insert({}, 1, sg0, "in0"), 0};
  auto c =
      m.insert(sg0, {sg1}, {{{in1, in0, 0}}}, {{{out0}}}, {}, {}, "call");
  const TensorId out1{c, 0};

  auto assertOrder = [](const auto &sched, const auto &a, const auto &b) {
    for (const auto &x : sched) {
      if (x == a.opId()) {
        return;
      }
      if (x == b.opId()) {
        throw poprithms::test::error("Order not satisifed");
      }
    }
  };

  {
    using namespace callstack_test;
    const auto obs = Querier(m).scheduled(Querier::DataDepOrder::Fwd,
                                          Querier::GraphDepOrder::TopDown);
    assertOrder(obs, in0, x01);
    assertOrder(obs, x01, out0);
    assertOrder(obs, in1, out1);
    assertOrder(obs, out1, out0);
    assertOrder(obs, in1, in0);
  }

  {
    using namespace callstack_test;
    const auto obs = Querier(m).scheduled(Querier::DataDepOrder::Bwd,
                                          Querier::GraphDepOrder::BottomUp);
    assertOrder(obs, x01, in0);
    assertOrder(obs, out0, x01);
    assertOrder(obs, out1, in1);
    assertOrder(obs, out0, out1);
    assertOrder(obs, in0, in1);
  }
}

} // namespace

int main() {
  testOnMultiGraphPathTo0();
  testNoPathToTarget0();
  testNestedFullStack0();
  testMultiGraphBack0();
  testMultiGraphBack1();
  testSwitch0();
  testRepeat0();
  testScheduling0();
  return 0;
}
