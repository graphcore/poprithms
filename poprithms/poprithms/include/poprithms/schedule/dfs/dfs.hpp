// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_DFS_GRAPH
#define POPRITHMS_SCHEDULE_DFS_GRAPH

#include <cstdint>
#include <vector>

namespace poprithms {
namespace schedule {
namespace dfs {

/** A post-order depth-first-search traversal of a directed graph
 *
 * \param edges The forward edges of the graph. In particular, edges[i].size()
 *              is the number of forward edges of node i.
 *
 * Examples:
 *  If edges is {{1},{2},{}}, representing the graph
 *          0 -> 1 -> 2
 *
 *  Then the returned serialization will be {2,1,0}.
 *
 *
 *  If edges is {{3},{3},{0,1},{}}, representing the graph
 *          2 -> 0
 *            \   \.
 *             1 -> 3
 *
 *  Then the returned serialization will either be {3,0,1,2} or {3,1,0,2}.
 *
 * If the directed graph has a cycle. a solution is still returned. For
 * example if the graph is {{1},{0}}, the either {0,1} or {1,0} is returned.
 * This property is used in, for example, partitioning a graph into strongly
 * connected components.
 *
 * */

using Edges = std::vector<std::vector<uint64_t>>;
std::vector<uint64_t> postOrder(const Edges &edges);

} // namespace dfs
} // namespace schedule
} // namespace poprithms

#endif
