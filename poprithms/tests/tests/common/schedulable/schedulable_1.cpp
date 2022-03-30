// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <testutil/common/schedulable/graph.hpp>

#include <poprithms/common/schedulable/additionalfwdedges.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using poprithms::test::error;

using namespace poprithms::common::schedulable_test;
using poprithms::common::multiout::RemovalEvents;
using poprithms::common::schedulable::NoAdditionalFwdEdges;

template <typename t>
std::ostream &operator<<(std::ostream &ost, const std::vector<t> &opids) {
  poprithms::util::append(ost, opids);
  return ost;
}

// insert 1 op, then remove it.
void removal0() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 1, gid, "in0");
  g.removeOp(in0, {{}}, "test0");
  if (g.nOps() != 0) {
    throw error("1 op added, 1 op removed. 1 - 1 = 0 ops should remain");
  }

  const auto events = g.removalEvents();
  if (events.size() != 1) {
    throw error(
        "1 event was removed, I expect exactly 1 element RemovalEvents. ");
  }
}

void checkPostRemovalCopies() {

  Graph g;
  const auto gid = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 1, gid, "in0");
  g.removeOp(in0, {{}}, "test");

  if (g.isLive(in0)) {
    throw error("in0 should not be live, it was deleted");
  }

  // copy constructor
  auto g0 = g;

  // copy constructor
  Graph goo(g0);

  // assignment operator
  goo = g0;

  // move constructor
  auto foo = std::move(g0);

  // move constructor
  Graph shrew(std::move(goo));

  if (foo != g || shrew != g) {
    throw error("Comparisons post removal failed");
  }
}

//  add, remove like a stack.
//
//  {} -> {0} -> {0,1} -> {0} -> {} -> {2} -> {2,3} -> {2} -> {}
void removal1() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  for (uint64_t i = 0; i < 2; ++i) {
    const auto in0 = g.insert({}, 1, gid, {});
    const auto in1 = g.insert({}, 1, gid, {});
    g.removeOp(in1, {{}}, {});
    g.removeOp(in0, {{}}, {});
  }
  if (g.nOps() != 0) {
    throw error("Added 2, removed 2, added 2, removed 2. Should be 0 left");
  }

  RemovalEvents expected{
      {{/*OpId of Op removed */ 1,
        {},
        /*number of ops added when this removal event happens */ 2,
        {}},
       {0, {}, 2, {}},
       {3, {}, 4, {}},
       {2, {}, 4, {}}}};

  if (g.removalEvents() != expected) {
    std::ostringstream oss;
    oss << "Expected the removal events to be \n"
        << expected << ", but observed \n"
        << g.removalEvents();
    throw error(oss.str());
  }
}

void removal2() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 1, gid, {});
  const auto in1 = g.insert({}, 1, gid, {});
  const auto add = g.insert({{in0, 0}, {in0, 0}}, 1, gid, {});
  const auto mul = g.insert({{in0, 0}, {in1, 0}}, 1, gid, {});
  const auto g0  = g;
  g.removeOp(in0, {TensorId(in1, 0)}, {});
  if (g.nOps() != 3 ||
      g.inTensorIds(add) != TensorIds({{in1, 0}, {in1, 0}}) ||
      g.inTensorIds(mul) != TensorIds({{in1, 0}, {in1, 0}})) {
    std::ostringstream oss;
    oss << "Expected inputs to add and mul to be in1, after in0 removed. "
        << "This with initial graph " << g0 << ", and final graph " << g
        << ", and removal events : " << g.removalEventsStr();
    throw error(oss.str());
  }

  g.assertSchedulableGraphCorrectness();
}

void catchBadOpId0() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 1, gid, {});
  const auto in1 = g.insert({}, 1, gid, {});
  const auto in2 = g.insert({}, 1, gid, {});
  // This is fine:
  g.vanillaSubSchedule({in0, in2, in1}, NoAdditionalFwdEdges());

  g.removeOp(in1, {TensorId(in0, 0)}, {});

  bool caught{false};
  try {
    g.vanillaSubSchedule({in0, in2, in1}, NoAdditionalFwdEdges());
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error where not-live op is "
                                 "passed to vanillaSubSchedule");
  }
}

void removal3() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 2, gid, {});
  g.insert({{in0, 0}, {in0, 0}, {in0, 0}}, 1, gid, "");
  if (g.nConsumptionIds({in0, 0}) != 3 || g.nConsumptionIds({in0, 1}) != 0) {
    throw error("Basic test of nConsumptionIds failed.");
  }
}

//
//       data             data
//   'a' ---> 'toRemove0' ---> 'b'    'subst'
//
//       ---> 'toRemove1' --->
//     control          control
//
//   The output of 'subst' replaces the output of 'toRemove':
//
//                   data
//   'a'     'subst' ---> 'b'
//
//   In this test, we consider 2 cases:
//   1) there is a contol dep 'a'->'toRemove1'->'b', which is transferred to
//      'subst'
//   2) There is no such control dependency, and so 'a' can go a wondering
//   once
//     'toRemove' is gone.
//
//   This is the logic we've implemented, but it not obvious what the best set
//   of rules for transferring control dependencies is.
//
void removal4() {

  for (bool withInitialControlDeps : {true, false}) {

    Graph g;
    const auto gid       = g.createSubGraphId("g0");
    const auto a         = g.insert({}, 1, gid, {});
    const auto toRemove0 = g.insert({{a, 0}}, 1, gid, {});
    const auto toRemove1 = g.insert({}, 0, gid, {});
    const auto b         = g.insert({{toRemove0, 0}}, 1, gid, {});
    const auto subst     = g.insert({}, 1, gid, {});

    if (withInitialControlDeps) {
      g.constraint(a, toRemove1);
      g.constraint(toRemove1, b);
    }
    g.propagateControlDependencies(
        toRemove1, Graph::ControlDependencyPropagationType::ConserveLocally);
    g.removeOp(toRemove1, {TensorId(subst, 0)}, "removal4");
    g.removeOp(toRemove0, {TensorId(subst, 0)}, "removal5");

    bool caught{false};
    try {
      g.constraint(b, a);
      auto foo = g.randomSchedule(1011, NoAdditionalFwdEdges());
    } catch (const poprithms::error::error &e) {
      caught = true;
    }
    if (caught && !withInitialControlDeps) {
      throw error("The insertion of b->a should NOT create a cycle");
    }
    if (!caught && withInitialControlDeps) {
      throw error("The insertion of b->a should create a cycle");
    }
  }
}

void compare0() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  g.insert({}, 1, gid, {});
  auto gc = g;

  if (gc != g) {
    throw error("Graphs are identical, failed comparison test");
  }
  const auto b = g.insert({}, 1, gid, {});
  if (gc == g) {
    throw error("at this point the graphs are not the same, not even same "
                "number of ops. failed test");
  }
  g.removeOp(b, {}, {});
  if (gc == g) {
    throw error("at this point the graphs are still not the same, even "
                "though they are the same DAG. removed ops leave a trace");
  }
}

void testAdditionalConstraints0() {
  // Graph with 10 ops, with no data or control deps. Constraints are provided
  // as AdditionalFwdEdges, pinning the schedule down completely.
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  OpIds ops;
  uint64_t N = 10;
  for (uint64_t i = 0; i < N; ++i) {
    ops.push_back(g.insert({}, 1, gid, {}));
  }

  std::vector<uint64_t> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 twister(1011);
  std::shuffle(order.begin(), order.end(), twister);
  std::map<OpId, OpIds> fwds;
  for (uint64_t i = 0; i < N - 1; ++i) {
    fwds.insert({order[i], {order[i + 1]}});
  }

  auto fm = poprithms::common::schedulable::AdditionalFwdEdgesFromMap(fwds);
  auto sched = g.randomSchedule(1011, fm);
  for (uint64_t i = 0; i < N; ++i) {
    if (sched[i] != order[i]) {
      throw error(
          "Failed to obtain correct order with additional constraints");
    }
  }

  if (g.hasUniqueSchedule(gid) || !g.hasUniqueSchedule(gid, fm)) {
    throw error("error of schedule uniqueness with additional constraints");
  }
}

void testConstraintPhobic0() {

  Graph g;
  const auto gid = g.createSubGraphId("g0");

  // initialization : no compute = constraint phobic.
  auto init0 = g.insert({}, 1, gid, "init0", true);

  // slice : view-change = constraint phobic.
  auto s0 = g.insert({{init0, 0}}, 1, gid, "slice0", true);
  auto s1 = g.insert({{init0, 0}}, 1, gid, "slice0", true);

  // add : does compute.
  g.insert({{s0, 0}, {s1, 0}}, 1, gid, "add", false);

  auto preControl = g.getForwardEdgeMap_u64().fwdEdgesCompact();

  g.constraint(s0, s1);
  auto withOneControl = g.getForwardEdgeMap_u64().fwdEdgesCompact();

  if (preControl != withOneControl) {
    throw error("Constraint from slice should be pushed off the start");
  }

  g.constraint(s1, s0);
  auto withTwoControl = g.getForwardEdgeMap_u64().fwdEdgesCompact();
  if (preControl != withTwoControl) {
    throw error(
        "Both constraints from slices should be pushed off the start");
  }
}

void testConstraintPhobic1() {

  Graph g;
  const auto gid = g.createSubGraphId("sg0");
  auto x0        = g.insert({}, 1, gid, "");
  auto x1        = g.insertPhobic({}, 1, gid, "");
  auto x2        = g.insertPhobic({}, 1, gid, "");
  auto x3        = g.insert({}, 1, gid, "");

  // x0 -> x1
  // x1 -> x2
  // x2 -> x3
  //
  // where x1 and x2 are constraint phobic, should become
  //
  // x0 -> x3.

  g.constraint(x0, x1);
  g.constraint(x1, x2);
  g.constraint(x2, x3);

  auto fm = g.getForwardEdgeMap_u64();
  if (fm.fwdEdgesCompact().at(fm.compactId(x0)) !=
      std::vector<uint64_t>{fm.compactId(x3)}) {
    throw error("Failed to transfer phobic constraints correctly");
  }
}

void testCycle0() {

  Graph g;
  const auto gid = g.createSubGraphId("g0");
  auto x0        = g.insert({}, 1, gid, "in");
  auto x1        = g.insert({{x0, 0}}, 1, gid, "x1");
  auto x2        = g.insert({}, 1, gid, "in2");
  g.constraint(x1, x2);

  bool caught{false};
  try {
    auto sched = g.vanillaSchedule(
        poprithms::common::schedulable::AdditionalFwdEdgesFromMap(
            std::map<OpId, OpIds>{{x2, {x0}}}));
  } catch (const poprithms::error::error &em) {
    std::string w(em.what());
    for (auto frag : {"Edge types", "data", "control", "additional"}) {
      if (w.find(frag) == std::string::npos) {
        std::ostringstream oss;
        oss << "Expected to find \"" << frag
            << "\" in the error message for the cycle. ";
        throw error(oss.str());
      }
    }
    caught = true;
  }
  if (!caught) {
    throw error("Failed to catch cycle");
  }

  // Error message looks like:
  //
  // Op (debug name)         Op (local id) Edge ends (local ids) Edge types
  // ---------------         ------------- --------------------- ----------
  // schedulable_test::Op::0 0             (1)                   (data)
  // schedulable_test::Op::1 1             (2)                   (control)
  // schedulable_test::Op::2 2             (0)                   (additional)
}

} // namespace

int main() {
  removal0();
  removal1();
  removal2();
  removal3();
  removal4();
  compare0();
  checkPostRemovalCopies();
  catchBadOpId0();
  testAdditionalConstraints0();
  testConstraintPhobic0();
  testConstraintPhobic1();
  testCycle0();

  return 0;
}
