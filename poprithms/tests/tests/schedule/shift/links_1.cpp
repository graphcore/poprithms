// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <vector>

#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {
using namespace poprithms::schedule::shift;

void test0() {

  // X -- X -- X
  // ======

  Graph g;
  auto alloc0 = g.insertAlloc(1.0f);
  auto ops    = g.insertOps({"op0", "op1", "op2"});
  g.insertLink(ops[0], ops[1]);
  g.insertConstraint(ops[1], ops[2]);
  g.insertOpAlloc(ops, alloc0);
  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    TransitiveClosureOptimizations::allOn(),
                    RotationTermination::preStart());

  if (sg.getSubSchedule(ops) != std::vector<OpAddress>{0, 1, 2}) {
    throw poprithms::test::error("Expected schedule to be {0,1,2}");
  }
}

void test1() {

  //         0
  //     /  /|\\ \    .
  //    /  / | \\ \   .
  //   1  2  3  4  5  .
  //    \ \\ | /  /   .
  //     \ \\|/  /    .
  //         6
  //
  // tie 0->4, 2->6. Expect {0,4,1,5,3,2,6}
  //                         ===       ===

  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3", "op4", "op5", "op6"});
  g.insertLink(ops[0], ops[4]);
  g.insertLink(ops[2], ops[6]);
  g.insertOpAlloc({ops[3], ops[6]}, g.insertAlloc(1000.0f));
  g.insertOpAlloc({ops[5], ops[6]}, g.insertAlloc(100.0f));
  for (uint64_t i = 1; i < 6; ++i) {
    g.insertConstraint(ops[0], ops[i]);
    g.insertConstraint(ops[i], ops[6]);
  }
  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    TransitiveClosureOptimizations::allOff(),
                    RotationTermination::preStart());
  if (sg.scheduleToOp(0) != 0 || sg.scheduleToOp(1) != 4 ||
      sg.scheduleToOp(5) != 2 || sg.scheduleToOp(6) != 6) {
    throw poprithms::test::error("Expected  4 tied to start and 2 to end");
  }
}

void test2() {

  //
  //    X
  //  /  \\   .
  // X     X  .
  //  \  //   .
  //    X
  //
  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[1], ops[3]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[2], ops[3]);
  if (ScheduledGraph::isSchedulable(g)) {
    throw poprithms::test::error(
        "Diamond with tight edge is not schedulable");
  }
}

void test3() {

  //
  //    X
  //  //  \   .
  // X     X  .
  //  \  //   .
  //    X
  //
  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[2], ops[3]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[1], ops[3]);
  if (!ScheduledGraph::isSchedulable(g)) {
    throw poprithms::test::error(
        "Diamond with separated tight edges is schedulable");
  }
}

void test4() {
  const uint64_t seed0 = 1011;

  auto g0 = getRandomGraph(200, 4, 13, seed0);
  auto g1 = g0;

  ScheduledGraph sg0(std::move(g0),
                     KahnTieBreaker::RANDOM,
                     TransitiveClosureOptimizations::allOff(),
                     RotationTermination::preStart());

  // We know the random graph includes no internal ops.
  const auto sched0 = sg0.viewInternalScheduleToOp();

  for (uint64_t i = 1; i < sched0.size(); ++i) {
    g1.insertLink(sched0[i - 1], sched0[i]);
  }
  ScheduledGraph sg1(std::move(g1),
                     KahnTieBreaker::RANDOM,
                     TransitiveClosureOptimizations::allOff(),
                     RotationTermination::preStart());

  if (sg1.viewInternalScheduleToOp() != sched0) {
    throw poprithms::test::error(
        "Expected that inserting links between all Ops in the "
        "initial schedule would result in the same schedule");
  }
}

} // namespace

int main() {
  std::cout << "test 0" << std::endl;
  test0();

  std::cout << "test 1" << std::endl;
  test1();

  std::cout << "test 2" << std::endl;
  test2();

  std::cout << "test 3" << std::endl;
  test3();

  std::cout << "test 4" << std::endl;
  test4();
  return 0;
}
