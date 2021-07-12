// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/opalloc.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>
#include <testutil/schedule/shift/bifurcate_generator.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

Graph getBifurcatingGraph0(uint64_t D) {

  Graph g;

  // the root "o" in the figure above
  auto in0Mm = g.insertAlloc(1);
  auto inOp  = g.insertOp({}, {in0Mm}, "o");

  auto getFwdSplit = [&g](OpAlloc oa) {
    auto op  = oa.op;
    auto mm  = oa.alloc;
    auto mm0 = g.insertAlloc(1);
    auto op0 =
        g.insertOp({op}, {mm0, mm}, g.getOp(op).getDebugString() + "0");
    auto mm1 = g.insertAlloc(1);
    auto op1 =
        g.insertOp({op}, {mm1, mm}, g.getOp(op).getDebugString() + "1");
    return std::array<OpAlloc, 2>{OpAlloc{op0, mm0}, OpAlloc{op1, mm1}};
  };

  auto getBwdTie = [&g](std::array<OpAlloc, 2> oas) {
    auto oa0  = std::get<0>(oas);
    auto oa1  = std::get<1>(oas);
    auto dbs0 = g.getOp(oa0.op).getDebugString();
    auto dbs  = "y" + dbs0.substr(1, dbs0.size() - 2);
    auto mm   = g.insertAlloc(1);
    auto op   = g.insertOp({oa0.op, oa1.op}, {oa0.alloc, oa1.alloc, mm}, dbs);
    return OpAlloc{op, mm};
  };

  std::vector<std::vector<OpAlloc>> xs;
  xs.push_back({{inOp, in0Mm}});
  while (xs.back().size() < (1 << D)) {
    std::vector<OpAlloc> newXs;
    for (auto x : xs.back()) {
      auto x2 = getFwdSplit(x);
      newXs.push_back(std::get<0>(x2));
      newXs.push_back(std::get<1>(x2));
    }
    xs.push_back(newXs);
  }

  while (xs.back().size() != 1UL) {
    std::vector<OpAlloc> newXs;
    for (uint64_t i = 0; i < xs.back().size() / 2; ++i) {
      auto iStart = 2 * i;
      newXs.push_back(getBwdTie(
          std::array<OpAlloc, 2>{xs.back()[iStart], xs.back()[iStart + 1]}));
    }
    xs.push_back(newXs);
  }

  g.insertOp({xs.back().back().op}, {xs.back().back().alloc}, "return");
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
