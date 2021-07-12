// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {

using namespace poprithms::schedule::shift;

poprithms::schedule::shift::ScheduledGraph
getGraph(bool with_4_7_edge, bool constrainWSGs, bool bigDrop_6 = false) {
  //
  //                0       .
  //               / \      .
  //              /    \    .
  //            1       2   .
  //           /\      / \  .
  //          3   4   5   6 .
  //          \  . \ /   /  .
  //           7    8   9   .
  //             \     /    .
  //               \  /     .
  //                10      .
  //
  // 1,3,4 and 7 all have big negative drops (~ -1000)
  // 10 has a gigantic negative drop (~ -100000)
  // As {1,3,4,7} all have bigger drops than {2,5,6,9}, we expect an edge to
  // be inserted: 1->2. Moreover, when there is an edge 4->7 in the initial
  // graph, we expect an edge 4->2 to be inserted too, as with the edge 4->7,
  // 8 and 10 (the only ops downstream of both 1 and 2) are also downstream
  // of 4.

  Graph g;
  for (uint64_t i = 0; i < 11; ++i) {
    g.insertOp("Op" + std::to_string(i));
  }
  g.insertConstraints({{0, 1},
                       {0, 2},
                       {1, 3},
                       {1, 4},
                       {2, 5},
                       {2, 6},
                       {3, 7},
                       {4, 8},
                       {5, 8},
                       {6, 9},
                       {7, 10},
                       {9, 10}});

  if (with_4_7_edge) {
    g.insertConstraint(4, 7);
  }

  for (uint64_t i = 0; i < 11; ++i) {
    auto allocId = g.insertAlloc(0.1);
    auto ops     = g.getOp(i).getOuts();
    ops.push_back(i);
    g.insertOpAlloc(ops, allocId);
  }

  for (OpAddress id : {1, 3, 4, 7}) {
    auto allocId = g.insertAlloc(1000);
    g.insertOpAlloc({0, id}, allocId);
  }

  auto alloc10 = g.insertAlloc(100000);
  g.insertOpAlloc({0, 10}, alloc10);

  if (bigDrop_6) {
    auto alloc6 = g.insertAlloc(100000);
    g.insertOpAlloc({0, 6}, alloc6);
  }

  auto tco = TransitiveClosureOptimizations::allOff().withMaxIterations(1);
  if (constrainWSGs) {
    tco.withConstrainWeightSeparatedGroups();
  }

  return ScheduledGraph(std::move(g),
                        KahnTieBreaker::RANDOM,
                        tco,
                        RotationTermination::preStart());
}
} // namespace

int main() {
  using namespace poprithms::schedule::shift;

  //  no optim:
  auto g00 = getGraph(0, 0);
  auto g10 = getGraph(1, 0);

  // optim:
  auto g01 = getGraph(0, 1);
  auto g11 = getGraph(1, 1);

  // big-drop 6:
  auto g111 = getGraph(1, 1, 1);

  for (OpAddress i = 0; i < 11; ++i) {
    if (i != 2) {
      if (g00.getOp(i).getIns() != g01.getOp(i).getIns() ||
          g10.getOp(i).getIns() != g11.getOp(i).getIns()) {
        throw poprithms::test::error(
            "Expected unchanged inputs for all but 2");
      }
    }
    if (g01.getOp(2).getIns() != std::vector<OpAddress>{0, 1}) {
      throw poprithms::test::error(
          "Expected 0 and 1 as inputs to 2 post-optimization (no edge)");
    }
    if (g11.getOp(2).getIns() != std::vector<OpAddress>{0, 1, 4}) {
      throw poprithms::test::error(
          "Expected 0,1 and 4 as inputs to 2 post-optimization (+edge)");
    }

    if (g111.getOp(2).getIns() != g00.getOp(2).getIns()) {
      throw poprithms::test::error(
          "Expected unchanged inputs with big cost drop on 6");
    }
  }
  return 0;
}
