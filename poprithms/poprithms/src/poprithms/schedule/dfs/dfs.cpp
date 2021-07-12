// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/dfs/error.hpp>

#include <poprithms/schedule/dfs/dfs.hpp>

namespace poprithms {
namespace schedule {
namespace dfs {

// Example:
//
//
// 0 - 1    7    9 .
//   \   \   \   .
//     3 - 2 - 6.
//
// S           schedule
// -------     ---------
// {}
// {0}
// {0,1}
// {0,1,2}
// {0,1,2,6}
// {0,1,2}     {6}
// {0,1}     {6,2}
// {0}     {6,2,1}
// {0,3}     {6,2,1}
// {0}     {6,2,1,3}
// {}     {6,2,1,3,0}
// {7}     {6,2,1,3,0}
// {}     {6,2,1,3,0,7}
// {}     {6,2,1,3,0,7}
// {9}     {6,2,1,3,0,7}
// {}     {6,2,1,3,0,7,9}
//

std::vector<uint64_t> postOrder(const Edges &edges) {

  const auto N = edges.size();

  // The number of out-edges traversed. When this equals the total number of
  // out-edges for a node, the node is scheduled.
  std::vector<uint64_t> nChildrenVisited(edges.size(), 0);

  // The stack, which ids only enter once.
  std::vector<uint64_t> S;

  // set to true when a node is placed on the stack.
  std::vector<bool> visited(N, false);

  std::vector<uint64_t> schedule;
  schedule.reserve(N);

  for (uint64_t i = 0; i < N; ++i) {
    if (!visited[i]) {
      S.push_back(i);
      visited[i] = true;
      while (!S.empty()) {
        const auto nxt = S.back();
        if (nChildrenVisited[nxt] == edges[nxt].size()) {

          // transfer from stack to schedule.
          schedule.push_back(nxt);
          S.pop_back();
        } else {
          const auto child = edges[nxt][nChildrenVisited[nxt]];
          if (!visited[child]) {
            S.push_back(child);
            visited[child] = true;
          }
          ++nChildrenVisited[nxt];
        }
      }
    }
  }
  return schedule;
}

} // namespace dfs
} // namespace schedule
} // namespace poprithms
