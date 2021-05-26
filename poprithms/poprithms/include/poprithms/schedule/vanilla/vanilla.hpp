// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_VANILLA_HPP
#define POPRITHMS_SCHEDULE_VANILLA_VANILLA_HPP

#include <vector>

namespace poprithms {
namespace schedule {
namespace vanilla {

enum class ErrorIfCycle {
  No = 0, ///< Do not error if there is a cycle, but return the partial
          ///< schedule of whatever could be scheduled.
  Yes     ///< Throw an error if there is a cycle.
};

enum class VerifyEdges {
  No = 0, ///< Do no checks that edges are valid
  Yes ///< Check that edges are valid, in particular, check that all edges end
      ///< at valid nodes.
};

template <typename T> using Edges = std::vector<std::vector<T>>;

std::vector<uint64_t>
getSchedule_u64(const Edges<uint64_t> &fwdEdges, ErrorIfCycle, VerifyEdges);

std::vector<int64_t>
getSchedule_i64(const Edges<int64_t> &fwdEdges, ErrorIfCycle, VerifyEdges);

/**
 * Return true if there is exactly 1 way to schedule the graph with forward
 * edges #fwdEdges. If there are zero (due to a cycle) or multiple ways to
 * schedule the graph, then false is returned.
 * */
bool hasUniqueSchedule_u64(const Edges<uint64_t> &fwdEdges, VerifyEdges);
bool hasUniqueSchedule_i64(const Edges<int64_t> &fwdEdges, VerifyEdges);

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
