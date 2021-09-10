// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>
#include <vector>

#include <testutil/common/schedulable/schedulablegraph.hpp>

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using poprithms::test::error;

using namespace poprithms::common::schedulable_test;
using poprithms::common::multiout::RemovalEvents;

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
  g.removeSchedulableOp(in0, {{}}, "test0");
  if (g.nOps() != 0) {
    throw error("1 op added, 1 op removed. 1 - 1 = 0 ops should remain");
  }

  const auto events = g.removalEvents();
  if (events.size() != 1) {
    throw error(
        "1 event was removed, I expect exactly 1 element RemovalEvents. ");
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
    g.removeSchedulableOp(in1, {{}}, {});
    g.removeSchedulableOp(in0, {{}}, {});
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
  g.removeSchedulableOp(in0, {TensorId(in1, 0)}, {});
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

void removal3() {
  Graph g;
  const auto gid = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 2, gid, {});
  g.insert({{in0, 0}, {in0, 0}, {in0, 0}}, 1, gid, "");
  if (g.nConsumptionIds({in0, 0}) != 3 || g.nConsumptionIds({in0, 1}) != 0) {
    throw error("Basic test of nConsumptionIds failed.");
  }
}

//   'a' --> toRemove -----> 'b'
//
//   'subst'
//
//
//   replacing toRemove with subst results in a new constraint 'a' -> 'b',
//   which might not always be desirable.
void removal4() {
  Graph g;
  const auto gid      = g.createSubGraphId("g0");
  const auto a        = g.insert({}, 1, gid, {});
  const auto toRemove = g.insert({{a, 0}}, 1, gid, {});
  const auto b        = g.insert({{toRemove, 0}}, 1, gid, {});
  const auto subst    = g.insert({}, 1, gid, {});
  g.removeSchedulableOp(toRemove, {TensorId(subst, 0)}, {});

  // 'subst' replaces 'toRemove'. Is 'a' --> 'b' present? We assert not but
  // inserting 'b' -> 'a' and detecting a cylce.

  bool caught{false};
  try {
    g.constraint(b, a);
    auto foo = g.randomSchedule(1011);
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw error("The insertion of b->a should create a cycle");
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
  g.removeSchedulableOp(b, {}, {});
  if (gc == g) {
    throw error("at this point the graphs are still not the same, even "
                "though they are the same DAG. removed ops leave a trace");
  }
}

} // namespace

int main() {
  removal0();
  removal1();
  removal2();
  removal3();
  removal4();
  compare0();
  return 0;
}
