// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/util/printiter.hpp>
namespace {

poprithms::schedule::shift::Graph

getGraph() {

  //               x0 [S, M, L]
  //                |
  //       +--------+
  //       |        |
  //      x1 [M]    x2 [S]       x3
  //                |             |
  //                x4 [L] <------+
  //
  // where allocs are
  // [S] small
  // [M] medium
  // [L] large.
  //
  //
  // optimal (sum minimizing) schedule is:
  //    x3 x0      x2     x4  x1
  //    [] [S,M,L] [M,L], [M] []
  //

  using namespace poprithms::schedule::shift;
  Graph g;
  std::vector<OpAddress> ops;
  std::vector<AllocAddress> allocs;
  for (uint64_t i = 0; i < 5; ++i) {
    auto istr = std::to_string(i);
    ops.push_back(g.insertOp("op_" + istr));
  }

  for (auto [i, j] :
       std::vector<std::tuple<int, int>>{{0, 1}, {0, 2}, {2, 4}, {3, 4}}) {
    g.insertConstraint(ops[i], ops[j]);
  }

  auto S = g.insertAlloc(10);
  auto M = g.insertAlloc(100);
  auto L = g.insertAlloc(1000);
  g.insertOpAlloc({ops[0], ops[1]}, M);
  g.insertOpAlloc({ops[0], ops[2]}, S);
  g.insertOpAlloc({ops[0], ops[4]}, L);

  return g;
}

template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &ts) {
  poprithms::util::append(ost, ts);
  return ost;
}
} // namespace

int main() {

  using namespace poprithms::schedule::shift;
  auto g = getGraph();

  const std::vector<OpAddress> expected({
      3,
      0,
      2,
      4,
      1,
  });

  for (TransitiveClosureOptimizations tco :
       {TransitiveClosureOptimizations::allOn(),
        TransitiveClosureOptimizations::allOff()}) {
    for (KahnTieBreaker tb : {KahnTieBreaker::FIFO, KahnTieBreaker::GREEDY}) {

      ScheduledGraph sg(Graph(g), {tb, {}}, tco);

      if (sg.viewInternalScheduleToOp() != expected) {

        std::ostringstream oss;
        oss << "Failed to obtain the optimal schedule. "
            << "Expected " << expected << " but observed "
            << sg.viewInternalScheduleToOp() << ". This with kahnTieBreaker"
            << tb << " and transitive closure optimizations:\n"
            << tco << ".";
        throw poprithms::test::error(oss.str());
      }
    }
  }
  return 0;
}
