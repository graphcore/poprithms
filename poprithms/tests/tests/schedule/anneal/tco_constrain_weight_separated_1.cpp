#include <iostream>
#include <vector>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

poprithms::schedule::anneal::Graph

getGraph() {

  //       0--6
  //      /|   \
  //    /  3    7
  //   1   |   /
  //   |   4  8
  //   2   |   \
  //   |   5    9
  //   |   |   /
  //    \  |  /
  //      10
  //

  using namespace poprithms::schedule::anneal;
  Graph g;
  for (uint64_t i = 0; i < 11; ++i) {
    g.insertOp("Op" + std::to_string(i));
  }
  g.insertConstraints({{0, 1},
                       {1, 2},
                       {2, 10},
                       {0, 3},
                       {3, 4},
                       {4, 5},
                       {5, 10},
                       {0, 6},
                       {6, 7},
                       {7, 8},
                       {8, 9},
                       {9, 10}});

  // small allocs created by each op, used by its outs
  for (uint64_t i = 0; i < 11; ++i) {
    auto allocId = g.insertAlloc(1);
    auto ops     = g.getOp(i).getOuts();
    ops.push_back(i);
    g.insertOpAlloc(ops, allocId);
  }

  // 1,2 : small reduction in liveness
  for (OpAddress id : {1, 2}) {
    auto allocId = g.insertAlloc(100);
    g.insertOpAlloc({0, id}, allocId);
  }

  // 3,4,5 : medium reduction in liveness
  for (OpAddress id : {3, 4, 5}) {
    auto allocId = g.insertAlloc(200);
    g.insertOpAlloc({0, id}, allocId);
  }

  // 6,7,8,9 : large reduction in liveness
  for (OpAddress id : {6, 7, 8, 9}) {
    auto allocId = g.insertAlloc(300);
    g.insertOpAlloc({0, id}, allocId);
  }

  auto tco = TransitiveClosureOptimizations::allOff()
                 .withConstrainWeightSeparatedGroups()
                 .withMaxIterations(1);

  g.initialize(KahnTieBreaker::RANDOM, 1011, tco);

  return g;
}

int main() {
  using namespace poprithms::schedule::anneal;
  auto g = getGraph();
  if (g.getOp(1).getIns() != std::vector<OpAddress>{0, 3, 4, 5, 6, 7, 8, 9}) {
    throw error("Expected all unconstrained w.r.t. 1 to point to it");
  }
  if (g.getOp(3).getIns() != std::vector<OpAddress>{0, 6, 7, 8, 9}) {
    throw error(
        "Expected all unconstrained w.r.t. 3 on the 6-branch to point to it");
  }
  return 0;
}
