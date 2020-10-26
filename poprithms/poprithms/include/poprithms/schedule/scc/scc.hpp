// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SCC_SCC_HPP
#define POPRITHMS_SCHEDULE_SCC_SCC_HPP

#include <cstdint>
#include <string>
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

enum class IncludeSingletons { No = 0, Yes };

/**
 * Summarize the Connected Components of a graph.
 *
 * \param edges The edges of the graph being summarized.
 *
 * \param debugStrings String assocated with nodes in the graph. In
 *                     particular, debugStrings[i] corresponds to the source
 *                     node of edges[i].
 *
 * \param sings Defines whether components with a single node should be
 *              included in the summary.
 * */
std::string getSummary(const FwdEdges &edges,
                       std::vector<std::string> &debugStrings,
                       IncludeSingletons sings);

} // namespace scc
} // namespace schedule
} // namespace poprithms

#endif
