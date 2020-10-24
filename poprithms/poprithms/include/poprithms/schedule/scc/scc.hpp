// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SCC_SCC_HPP
#define POPRITHMS_SCHEDULE_SCC_SCC_HPP

#include <cstdint>
#include <vector>

namespace poprithms {
namespace schedule {
namespace scc {

using SCC      = std::vector<uint64_t>;
using SCCs     = std::vector<SCC>;
using FwdEdges = std::vector<std::vector<uint64_t>>;

/**
 * Get the Strongly Connected Components of a directed graph.
 *
 * Components are returned in topological order.
 *
 * See https://en.wikipedia.org/wiki/Strongly_connected_component
 *
 * Implementation based on the algorithm described by Dasgupta
 * et al, in Algorithms (2006).
 * */
SCCs getStronglyConnectedComponents(const FwdEdges &edges);

} // namespace scc
} // namespace schedule
} // namespace poprithms

#endif
