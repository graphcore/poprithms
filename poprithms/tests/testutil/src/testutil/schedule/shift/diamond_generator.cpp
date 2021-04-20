// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>
#include <testutil/schedule/shift/diamond_generator.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

//      ---<--x-->---
//      |    / \    |
//      x x x x x x x (the N intermediate Ops)
//      |    \ /    |
//      -->---x--<---

poprithms::schedule::shift::Graph getDiamondGraph0(uint64_t N) {

  using namespace poprithms::schedule::shift;

  Graph graph;

  auto root = graph.insertOp("root");
  auto tail = graph.insertOp("tail");

  for (int i = 0; i < N; ++i) {
    // weight decreases in N, so we expect ops with low addresses
    // (i.e. yhose with heavy weights) to be scheduled first
    auto op = graph.insertOp("op" + std::to_string(i));

    double w0 = N + 1 - i;
    auto a0   = graph.insertAlloc(w0);
    graph.insertOpAlloc({op, root}, a0);

    double w1 = 5;
    auto a1   = graph.insertAlloc(w1);
    graph.insertOpAlloc({op, tail}, a1);

    graph.insertConstraint(root, op);
    graph.insertConstraint(op, tail);
  }
  return graph;
}

void assertGlobalMinimumDiamondGraph0(const ScheduledGraph &graph,
                                      uint64_t N) {

  std::vector<OpAddress> expected;
  // the root:
  expected.push_back(0);
  for (int i = 0; i < N; ++i) {
    expected.push_back(i + 2);
  }
  // the tail:
  expected.push_back(1);

  for (ScheduleIndex i = 0; i < graph.nOps(); ++i) {
    if (graph.scheduleToOp(i) != expected[i]) {
      throw error("unexpected schedule in assertGlobalMinimumDiamondGraph0");
    }
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
