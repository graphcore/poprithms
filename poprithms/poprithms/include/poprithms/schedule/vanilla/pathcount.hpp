// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_SCHEDULE_VANILLA_PATHCOUNT_HPP
#define POPRITHMS_SCHEDULE_VANILLA_PATHCOUNT_HPP

#include <array>
#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/schedule/vanilla/types.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

/**
 *
 * Compute statistics for nodes in a DAG by accumulating values from output
 * nodes.
 *
 * Example:
 *
 * edges are
 * a -> {c}
 * b -> {c,d}
 * c -> {d,e}
 * d -> {}
 * e -> {}
 *
 *  a     +--<--b
 *  |     |     |
 *  +--c--+     v
 *     |        |
 *     +-->-+   |
 *     |    |   |
 *     |    +-+-+
 *     e      |
 *            d
 * Counts are,
 *  with Add: e:1 d:1 c:2 b:3 a:2
 *  with Max: e:1 d:1 c:1 b:2 a:2
 *  with Min: e:1 d:1 c:1 b:1 a:2
 *
 * */

enum class CountType {
  Add, ///< Count the number of paths which end at a terminal node.
  Max, ///< Measure the longest path which ends at a terminal node.
  Min  ///< Measure the shortest path which ends at a terminal node.
};

std::ostream &operator<<(std::ostream &, CountType);

class PathCounter {
public:
  /**
   * For each node in the graph with forward edges #fwdEdges, obtain the
   * statistic #ct. The graph is expected to be a DAG, this can be optionally
   * verified with flags #eic and #ve.
   * */
  static std::vector<uint64_t> count(const Edges<uint64_t> &fwdEdges,
                                     CountType,
                                     ErrorIfCycle,
                                     VerifyEdges);

  static std::vector<uint64_t>
  longestPathsToTerminal(const Edges<uint64_t> &fwdEdges,
                         ErrorIfCycle eic,
                         VerifyEdges ve) {
    return count(fwdEdges, CountType::Max, eic, ve);
  }

  static std::vector<uint64_t>
  shortestPathsToTerminal(const Edges<uint64_t> &fwdEdges,
                          ErrorIfCycle eic,
                          VerifyEdges ve) {
    return count(fwdEdges, CountType::Min, eic, ve);
  }
};

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
