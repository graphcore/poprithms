#include <iostream>
#include <testutil/schedule/anneal/randomgraph.hpp>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {
using namespace poprithms::schedule::anneal;

void test0() {

  // X -- X -- X
  // ======

  Graph g;
  auto alloc0 = g.insertAlloc(1.0f);
  auto ops    = g.insertOps({"op0", "op1", "op2"});
  g.insertLink(ops[0], ops[1]);
  g.insertConstraint(ops[1], ops[2]);
  g.insertOpAlloc(ops, alloc0);
  g.initialize(KahnTieBreaker::RANDOM, 1011);
  if (g.getScheduleToOp() != std::vector<OpAddress>{0, 1, 2}) {
    throw error("Expected schedule to be {0,1,2}");
  }
}

void test1() {

  //         X
  //     / / | \ \
  //    . .  .  . .
  //     \ \ | / /
  //         X
  //
  // tie begin to 4, end to 2. Expect {0,4...2,6}

  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3", "op4", "op5", "op6"});
  g.insertLink(ops[0], ops[4]);
  g.insertLink(ops[2], ops[6]);
  g.insertOpAlloc({ops[3], ops[6]}, g.insertAlloc(1000.0f));
  for (uint64_t i = 1; i < 6; ++i) {
    g.insertConstraint(ops[0], ops[i]);
    g.insertConstraint(ops[i], ops[6]);
  }
  g.initialize(KahnTieBreaker::RANDOM, 1011);
  if (g.scheduleToOp(0) != 0 || g.scheduleToOp(1) != 4 ||
      g.scheduleToOp(5) != 2 || g.scheduleToOp(6) != 6) {
    throw error("Expected  4 tied to start and 2 to end");
  }
}

void test2() {

  //
  //    X
  //  /  \\
  // X     X
  //  \  //
  //    X
  //
  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[1], ops[3]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[2], ops[3]);
  g.finalize();
  if (g.isSchedulable()) {
    throw error("Diamond with tight edge is not schedulable");
  }
}

void test3() {

  //
  //    X
  //  //  \
  // X     X
  //  \  //
  //    X
  //
  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[2], ops[3]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[1], ops[3]);
  g.finalize();
  if (!g.isSchedulable()) {
    throw error("Diamond with separated tight edges is schedulable");
  }
}

void test4() {
  const uint64_t seed0 = 1011;
  const uint64_t seed1 = 1012;
  const uint64_t seed2 = 1013;

  auto g0 = getRandomGraph(200, 4, 13, seed0);
  auto g1 = g0;

  g0.initialize(KahnTieBreaker::RANDOM, seed1);
  const auto &sched0 = g0.getScheduleToOp();

  for (ScheduleIndex i = 1; i < sched0.size(); ++i) {
    g1.insertLink(sched0[i - 1], sched0[i]);
  }
  g1.initialize(KahnTieBreaker::RANDOM, seed2);
  if (g1.getScheduleToOp() != sched0) {
    throw error("Expected that inserting links between all Ops in the "
                "initial schedule would result in the same schedule");
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  return 0;
}
