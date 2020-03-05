#ifndef POPRITHMS_SCHEDULE_DFS_GRAPH
#define POPRITHMS_SCHEDULE_DFS_GRAPH

#include <vector>

namespace poprithms {
namespace schedule {
namespace dfs {

// A post-order depth-first-search traversal of a directed graph
using Edges = std::vector<std::vector<uint64_t>>;
std::vector<uint64_t> postOrder(const Edges &edges);

} // namespace dfs
} // namespace schedule
} // namespace poprithms

#endif
