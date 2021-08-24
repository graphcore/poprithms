// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

poprithms::schedule::shift::ScheduledGraph

getGraph() {

  //
  //
  //
  //
  //   1 <- 2 <- 7---6
  //   |    |    |   |
  //   |    |    |   |
  //   |    3    4---5
  //   |    |   /
  //   |    |  /
  //  root (0)
  //
  //
  // id   lower-bound = upper-bound
  // 0   |  +1
  // 1   |  -2
  // 2   |  -1
  // 3   |  +1
  // 4   |  +1
  // 5   |   0
  // 6   |   0
  // 7   |  -1
  //
  // considering the case in constrainWeightSeparatedGroups of 4->3.
  // It should be inserted by the tie-breaker.

  using namespace poprithms::schedule::shift;
  constexpr uint64_t nOps{12};
  Graph g;
  for (uint64_t i = 0; i < nOps; ++i) {
    g.insertOp("Op" + std::to_string(i));
  }
  g.insertConstraints({{0, 1},
                       {0, 3},
                       {0, 4},
                       {2, 1},
                       {3, 2},
                       {4, 5},
                       {4, 7},
                       {5, 6},
                       {6, 7},
                       {7, 2}});

  // small allocs created by each op, used by its outs
  for (uint64_t i = 0; i < nOps; ++i) {
    auto allocId = g.insertAlloc(1);
    auto ops     = g.getOp(i).getOuts();
    ops.push_back(i);
    g.insertOpAlloc(ops, allocId);
  }

  auto tco = TransitiveClosureOptimizations::allOff()
                 .withConstrainWeightSeparatedGroups(true)
                 .withMaxIterations(1);

  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    tco,
                    RotationTermination::preStart());

  return sg;
}

int main() {
  using namespace poprithms::schedule::shift;
  auto g    = getGraph();
  auto outs = g.getOp(4).getOuts();
  if (std::find(outs.cbegin(), outs.cend(), 3) == outs.cend()) {
    throw poprithms::test::error("Expected 3 to be inserted as output of 4");
  }
  return 0;
}
