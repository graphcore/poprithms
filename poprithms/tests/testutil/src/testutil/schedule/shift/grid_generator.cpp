// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <testutil/schedule/shift/grid_generator.hpp>
#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/opalloc.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

poprithms::schedule::shift::Graph getGridGraph0(uint64_t N,
                                                uint64_t allocLower,
                                                uint64_t allocUpper,
                                                uint32_t seed) {
  using namespace poprithms::schedule::shift;
  Graph g;

  std::vector<std::vector<OpAddress>> grid;

  auto getName = [](int row, int col) {
    return std::to_string(row) + '_' + std::to_string(col);
  };

  for (int row = 0; row < N; ++row) {

    // the left column of "o"s in the figure above
    auto mm = g.insertAlloc(2 * N);
    std::vector<OpAddress> prods{};
    if (row != 0) {
      prods.push_back(grid.back().back());
    }
    auto op = g.insertOp(prods, std::vector<AllocAddress>{}, getName(row, 0));
    grid.push_back({{op, mm}});
  }

  for (int row = 0; row < N; ++row) {

    // the internal columns of the figure in the header file.
    for (int col = 1; col < N - 1; ++col) {
      auto mmSize = col == N / 2 ? 1 : 2 * N;
      auto mm     = g.insertAlloc(mmSize);
      auto op     = g.insertOp(std::vector<OpAddress>{grid[row].back()},
                           std::vector<AllocAddress>{},
                           getName(row, col));
      grid[row].push_back(op);
    }
  }

  // the rightmost column of the figure above
  for (int row = N - 1; row >= 0; --row) {

    auto mm = g.insertAlloc(2 * N);
    std::vector<OpAddress> prods{grid[row].back()};
    std::vector<AllocAddress> allocs{};
    if (row != N - 1) {
      prods.push_back(grid[row + 1].back());
    }
    auto op = g.insertOp(prods, allocs, getName(row, N - 1));
    grid[row].push_back(op);
  }

  addConnectedAllocs(g, allocLower, allocUpper, seed);

  return g;
}

void assertGlobalMinimumGridGraph0(const ScheduledGraph &g, uint64_t N) {
  AllocWeight expect(3 * 2 * N + (N - 2) * 1, 0);
  if (g.getMaxLiveness() != expect) {
    std::ostringstream oss;
    oss << "In assertGlobalMinimumGridGraph0, g.getMaxLiveness() gives "
        << g.getMaxLiveness() << " but "
        << "expected final max liveness to be 2*2*N + (N-2)*1 = " << expect;
    throw poprithms::test::error(oss.str());
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
