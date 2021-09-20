// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <testutil/schedule/shift/bifurcate_generator.hpp>
#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/opalloc.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

Graph getBifurcatingGraph0(uint64_t D,
                           uint64_t allocLower,
                           uint64_t allocUpper,
                           uint32_t seed) {

  Graph g;

  // the root "o" in the figure in the function declaration.
  auto inOp = g.insertOp({}, {}, "o");

  auto getFwdSplit = [&g](OpAddress op) {
    auto op0 = g.insertOp({op}, {}, g.getOp(op).getDebugString() + "0");
    auto op1 = g.insertOp({op}, {}, g.getOp(op).getDebugString() + "1");
    return std::array<OpAddress, 2>{op0, op1};
  };

  auto getBwdTie = [&g](std::array<OpAddress, 2> oas) {
    auto oa0  = std::get<0>(oas);
    auto oa1  = std::get<1>(oas);
    auto dbs0 = g.getOp(oa0).getDebugString();
    auto dbs  = "y" + dbs0.substr(1, dbs0.size() - 2);
    auto op   = g.insertOp({oa0, oa1}, {}, dbs);
    return op;
  };

  std::vector<std::vector<OpAddress>> xs;
  xs.push_back({inOp});
  while (xs.back().size() < (1 << D)) {
    std::vector<OpAddress> newXs;
    for (auto x : xs.back()) {
      auto x2 = getFwdSplit(x);
      newXs.push_back(std::get<0>(x2));
      newXs.push_back(std::get<1>(x2));
    }
    xs.push_back(newXs);
  }

  while (xs.back().size() != 1UL) {
    std::vector<OpAddress> newXs;
    for (uint64_t i = 0; i < xs.back().size() / 2; ++i) {
      auto iStart = 2 * i;
      newXs.push_back(getBwdTie(std::array<OpAddress, 2>{
          xs.back()[iStart], xs.back()[iStart + 1]}));
    }
    xs.push_back(newXs);
  }

  g.insertOp({xs.back().back()}, {}, "return");

  addConnectedAllocs(g, allocLower, allocUpper, seed);

  return g;
}

// Final max liveness should be D+2
void assertGlobalMinimumBifurcatingGraph0(const ScheduledGraph &g,
                                          uint64_t D) {
  auto finalMaxLiveness = g.getMaxLiveness();
  if (finalMaxLiveness != AllocWeight(D + 2, 0)) {
    throw poprithms::test::error("expected final max liveness to be D + 2");
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
