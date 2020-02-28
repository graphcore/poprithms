#include <poprithms/schedule/anneal/annealusings.hpp>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>
#include <testutil/schedule/anneal/grid_generator.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::schedule::anneal::Graph getGridGraph0(uint64_t N) {
  using namespace poprithms::schedule::anneal;
  Graph g;

  std::vector<std::vector<OpAlloc>> grid;

  auto getName = [](int row, int col) {
    return std::to_string(row) + '_' + std::to_string(col);
  };

  for (int row = 0; row < N; ++row) {

    // the left column of "o"s in the figure above
    auto mm = g.insertAlloc(2 * N);
    std::vector<OpAddress> prods{};
    std::vector<AllocAddress> allocs{mm};
    if (row != 0) {
      prods.push_back(grid.back().back().op);
      allocs.push_back(grid.back().back().alloc);
    }
    auto op = g.insertOp(prods, allocs, getName(row, 0));
    grid.push_back({{op, mm}});
  }

  for (int row = 0; row < N; ++row) {

    // the internal columns of the figure above
    for (int col = 1; col < N - 1; ++col) {
      auto mmSize = col == N / 2 ? 1 : 2 * N;
      auto mm     = g.insertAlloc(mmSize);
      auto op     = g.insertOp({grid[row].back().op},
                           {grid[row].back().alloc, mm},
                           getName(row, col));
      grid[row].push_back({op, mm});
    }
  }

  // the rightmost column of the figure above
  for (int row = N - 1; row >= 0; --row) {

    auto mm = g.insertAlloc(2 * N);
    std::vector<OpAddress> prods{grid[row].back().op};
    std::vector<AllocAddress> allocs{mm, grid[row].back().alloc};
    if (row != N - 1) {
      prods.push_back(grid[row + 1].back().op);
      allocs.push_back(grid[row + 1].back().alloc);
    }
    auto op = g.insertOp(prods, allocs, getName(row, N - 1));
    grid[row].push_back({op, mm});
  }

  return g;
}

void assertGlobalMinimumGridGraph0(const Graph &g, uint64_t N) {
  AllocWeight expect(3 * 2 * N + (N - 2) * 1, 0);
  if (g.getMaxLiveness() != expect) {
    std::ostringstream oss;
    oss << "In assertGlobalMinumumGridGraph0, g.getMaxLiveness() gives "
        << g.getMaxLiveness() << " but "
        << "expected final max liveness to be 2*2*N + (N-2)*1 = " << expect;
    throw error(oss.str());
  }
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
