// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;
  AnnealCommandLineOptions annopts;
  auto opts = annopts.getCommandLineOptionsMap(
      argc,
      argv,
      {"N"},
      {"The number of rows/cols in the grid (the number of nodes is N**2)"});
  auto N = std::stoi(opts.at("N"));

  //                                         (N-1, N-1)
  //      o  ->  o  ->  o  ->  z  ->  o  ->  o
  //      ^                    =             |
  //      |                                 \ /
  //      o  ->  o  ->  o  ->  z  ->  o  ->  o ====== most expensive point:
  //      ^                    =             |        3 expensives are live
  //      |                                 \ /       N-2 cheaps are live.
  //      o  ->  o  ->  o  ->  z  ->  o  ->  o
  //      ^                    =             |
  //      |                                 \ /
  //      o  ->  o  ->  o  ->  z  ->  o  ->  o
  //      ^                    =             |
  //      |                                 \ /
  //      o  ->  o  ->  o  ->  z  ->  o  ->  o
  //      ^                    =             |
  //      |                                 \ /
  //      o  ->  o  ->  o  ->  z  ->  o  ->  o
  //  (0,0)                    =
  //
  //  An N x N grid of ops resembling forwards-backwards of an nn.
  //  @z alloc is of size 1 @o alloc is of size 2*N
  //
  //  max should be in [3*2*N + (N-2)*1, O(N^2)]

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

  std::cout << g << std::endl;

  // set schedule and all related variables
  g.initialize();
  std::cout << g.getLivenessString() << std::endl;

  g.minSumLivenessAnneal(annopts.getAlgoCommandLineOptionsMap(opts));

  std::cout << g.getLivenessString() << std::endl;

  AllocWeight expect(3 * 2 * N + (N - 2) * 1, 0);
  if (g.getMaxLiveness() != expect) {
    std::ostringstream oss;
    oss << "getMaxLiveness() gives " << g.getMaxLiveness() << " but "
        << "expected final max liveness to be 2*2*N + (N-2)*1 = " << expect;
    throw poprithms::schedule::anneal::error(oss.str());
  }

  return 0;
}
