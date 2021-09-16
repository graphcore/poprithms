// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <vector>

#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {
using namespace poprithms::schedule::shift;

void test0() {

  // The linked diamond,
  //
  //    X0    .
  //  //  \   .
  // X1    X2 .
  //  \  //   .
  //    X3    .
  //
  // but with allocs which would prefer the mirror-image linkage. Are the
  // links preserved?

  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[2], ops[3]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[1], ops[3]);

  // Allocs want to go against the links:
  auto alloc0 = g.insertAlloc(100.0f);
  g.insertAlloc(100.0f);
  g.insertOpAlloc({ops[0], ops[2]}, alloc0);
  g.insertOpAlloc({ops[1], ops[3]}, alloc0);

  ScheduledGraph sg(std::move(g), KahnTieBreaker::RANDOM);

  if (sg.scheduleToOp(0) != 0 || sg.scheduleToOp(1) != 1 ||
      sg.scheduleToOp(2) != 2 || sg.scheduleToOp(3) != 3) {
    throw poprithms::test::error("The Links between Ops are not satisfied");
  }
}

void test1() {

  //
  // A random test that links are preserved
  //

  int seed0 = 1011;
  auto nOps = 120;
  auto g0   = getRandomGraph(nOps, 3, 7, seed0);
  auto g1   = g0;

  auto initializeGraph = [](Graph &g) {
    ScheduledGraph sg(Graph(g),
                      KahnTieBreaker::RANDOM,
                      TransitiveClosureOptimizations::allOff(),
                      RotationTermination::nHours(1),
                      RotationAlgo::RIPPLE,
                      1011,
                      FileWriter::None(),
                      DebugMode::On);
    return sg;
  };

  ScheduledGraph sgHalfBaked(Graph(g0),
                             KahnTieBreaker::RANDOM,
                             TransitiveClosureOptimizations::allOff(),
                             RotationTermination::preStart());

  // g1 is like g0, but with a few Links inserted
  for (ScheduleIndex i = 0; i < nOps - 1; ++i) {
    if (i % 3 == 0) {
      g1.insertLink(sgHalfBaked.scheduleToOp(i),
                    sgHalfBaked.scheduleToOp(i + 1));
    }
  }

  const auto sg0 = initializeGraph(g0);
  const auto sg1 = initializeGraph(g1);

  // 1) confirm that Links are all satisified
  for (ScheduleIndex i = 0; i < nOps - 1; ++i) {
    const auto &op0 = sg1.getOp(sg1.scheduleToOp(i));
    if (op0.hasForwardLink()) {
      auto op1Address = op0.getForwardLink();
      if (sg1.opToSchedule(op1Address) != i + 1) {
        throw poprithms::test::error("Link is not satisfied");
      }
    }
  }

  auto linkLessSum = sg0.getSumLiveness();
  auto linkedSum   = sg1.getSumLiveness();
  std::cout << "Link-less energy : " << linkLessSum << std::endl;
  std::cout << "With-link energy : " << linkedSum << std::endl;
  if (linkLessSum >= linkedSum) {
    throw poprithms::test::error(
        "That is (very) odd, random links in a random graph result "
        "in a lower shifted liveness sum");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
