// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <random>

#include <poprithms/schedule/dfs/dfs.hpp>
#include <poprithms/schedule/dfs/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::dfs;

std::ostream &operator<<(std::ostream &os, const std::vector<uint64_t> &x) {
  poprithms::util::append(os, x);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Edges &x) {
  os << "\n    edges\n-------------";
  int cnt = 0;
  for (const auto &e : x) {
    os << "\n     " << cnt << ":";
    poprithms::util::append(os, e);
  }
  return os;
}

// Check that all ids in edges[i] are scheduled before i:
void assertCorrect(const Edges &edges) {
  const auto schedule = postOrder(edges);
  for (uint64_t scheduleIndex = 0; scheduleIndex < edges.size();
       ++scheduleIndex) {
    const auto nodeId = schedule[scheduleIndex];
    for (const auto outId : edges[nodeId]) {
      const auto iterEnd = std::next(schedule.cbegin(), scheduleIndex);
      if (std::find(schedule.cbegin(), iterEnd, outId) == iterEnd) {
        std::ostringstream oss;
        oss << "Failure in postOrder test. This for " << edges << "."
            << "\n\nThe obtained \"schedule\" is: \n    " << schedule
            << ". \nThis is not valid, as "
            << "the node " << nodeId << ", which appears at index "
            << schedule[scheduleIndex] << ", appears before " << outId << ".";
        throw error(oss.str());
      }
    }
  }
}

} // namespace

int main() {

  // 0 -> 1 -> 2 -> 3
  assertCorrect({{1}, {2}, {3}, {}});

  // 0 -> 1 -> 2 -> 3
  // 4 -> 5 -> 6 -> 7
  assertCorrect({{1}, {2}, {3}, {}, {5}, {6}, {7}, {}});

  // 0 -> 1 -> 2 -> 3
  //        \    /
  // 4 -> 5 -> 6 -> 7
  assertCorrect({{1}, {2, 6}, {3}, {}, {5}, {6}, {7, 3}, {}});

  // 0 -> 1 -> 2 -> 3
  //        \    /
  // 4 -> 5 -> 6 -> 7
  //     /
  // 8  /
  //   /
  // 12 -> 11 -> 10
  //  \        /
  //    9 - 13
  //
  //
  // 14
  //
  assertCorrect({
      {1},    // 0
      {2, 6}, // 1
      {3},    // 2
      {},     // 3
      {5},    // 4
      {6},    // 5
      {7, 3}, // 6
      {},     // 7
      {},     // 8
      {13},   // 9
      {},     // 10
      {10},   // 11
      {11},   // 12
      {10},   // 13
      {}      // 14
  });

  // Random graph tests:
  std::mt19937 gen(1011);
  auto getRandom =
      [&gen](uint64_t nNodes, uint64_t edgesPerNode, uint64_t range) {
        Edges edges(nNodes);
        std::vector<uint64_t> inds(range);
        std::iota(inds.begin(), inds.end(), 1);
        for (uint64_t i = 0; i < nNodes; ++i) {
          std::vector<uint64_t> deltas;
          std::sample(inds.cbegin(),
                      inds.cend(),
                      std::back_inserter(deltas),
                      edgesPerNode,
                      gen);
          for (auto delta : deltas) {
            const auto out = i + delta;
            if (out < edges.size()) {
              edges[i].push_back(out);
            }
          }
        }
        return edges;
      };
  for (uint64_t i = 0; i < 32; ++i) {
    assertCorrect(getRandom(200, 20, 40));
    assertCorrect(getRandom(200, 2, 40));
  }

  // Check that cycles are permitted:
  auto foo = postOrder({{1}, {2}, {0}});
  if (foo.size() != 3) {
    throw error("Failed to process graph with cycle (1)");
  }

  // Check a fully connected graph:
  foo = postOrder(
      {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4}, {0, 1, 2, 3}});
  if (foo.size() != 5) {
    throw error("Failed to process graph with cycle (2)");
  }

  return 0;
}
