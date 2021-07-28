// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <testutil/schedule/base/randomdag.hpp>

namespace poprithms {
namespace schedule {
namespace baseutil {

std::vector<std::vector<uint64_t>> randomConnectedDagToFinal(uint64_t N,
                                                             uint32_t seed) {

  std::mt19937 gen(seed);

  // forward and backward edges. initially empty.
  std::vector<std::vector<uint64_t>> fwd(N);
  std::vector<std::vector<uint64_t>> bwd(N);

  // is there a path (not necessarily direct) to the final node? Initially,
  // only if the node IS the final node, as initially there are no edges.
  std::vector<bool> isPath(N, false);
  isPath[N - 1] = true;

  // how many nodes have a path to the final node?
  // initially just 1, the final node.
  // We well add edges randomly until this is N.
  uint64_t nPaths{1};

  // depth first search, filling in where there is a path to the final node.
  auto flowBack = [&isPath, &nPaths, &bwd](uint64_t x) {
    std::vector<uint64_t> toProcess{x};
    std::unordered_set<uint64_t> visited;
    while (!toProcess.empty()) {
      auto nxt = toProcess.back();
      toProcess.pop_back();
      if (!isPath[nxt]) {
        isPath[nxt] = true;
        ++nPaths;
        for (auto src : bwd[nxt]) {
          toProcess.push_back(src);
          visited.insert(src);
        }
      }
    }
  };

  while (nPaths < N) {

    // generate a random edge a->b, 0 <= a < b < N.
    auto a = gen() % N;
    auto b = gen() % (N - 1);
    b += (b >= a ? 1 : 0);
    if (a > b) {
      std::swap(a, b);
    }

    // if it's a new edges, insert it.
    if (std::find(fwd[a].cbegin(), fwd[a].cend(), b) == fwd[a].cend()) {
      if (!isPath[a]) {
        fwd[a].push_back(b);
        bwd[b].push_back(a);

        // if moreover there's a path from b to the end, but not from a, then
        // by adding a->b we've created a path from a to the end. Register
        // this, and also register that there is a path for any node which
        // already has a path a.
        if (isPath[b] && !isPath[a]) {
          flowBack(a);
        }
      }
    }
  }

  return fwd;
}

std::vector<std::vector<uint64_t>> randomConnectedDag(uint64_t N,
                                                      uint32_t seed) {

  // If there are no nodes in the DAG, return the unique solution.
  if (N == 0) {
    return {};
  }

  // We first build a bidirectional connected graph, initialized with no
  // edges.
  std::vector<std::vector<uint64_t>> bidir(N);

  // Which nodes are connected to node 0? We'll add edges randomly until all
  // nodes are connected to node 0.
  std::vector<bool> connectedToNode0(N, false);
  connectedToNode0[0] = true;
  uint64_t nConnectedToNode0{1};

  std::mt19937 gen(seed);

  // starting from node 'i' which is connected to '0', perform a depth-first
  // search to find any nodes which are newly connected to node '0'.
  const auto update =
      [&bidir, &connectedToNode0, &nConnectedToNode0](uint64_t i) {
        std::vector<uint64_t> toProcess{i};
        while (!toProcess.empty()) {
          auto nxt = toProcess.back();
          toProcess.pop_back();

          // For all neighbors, if neightbor is already known to be connected
          // to node '0', do nothing. Otherwise, register its connection and
          // continue the search.
          for (auto node : bidir[nxt]) {
            if (!connectedToNode0[node]) {
              connectedToNode0[node] = true;
              ++nConnectedToNode0;
              toProcess.push_back(node);
            }
          }
        }
      };

  while (nConnectedToNode0 < N) {
    auto a = gen() % N;
    auto b = gen() % N;
    if (a != b &&
        std::find(bidir[a].cbegin(), bidir[a].cend(), b) == bidir[a].cend()) {
      bidir[a].push_back(b);
      bidir[b].push_back(a);
      if (connectedToNode0[a] && !connectedToNode0[b]) {
        update(a);
      } else if (!connectedToNode0[a] && connectedToNode0[b]) {
        update(b);
      }
    }
  }

  std::vector<std::vector<uint64_t>> fwds(N);
  for (uint64_t i = 0; i < N; ++i) {
    for (auto x : bidir[i]) {
      if (x > i) {
        fwds[i].push_back(x);
      }
    }
  }
  return fwds;
}

} // namespace baseutil
} // namespace schedule
} // namespace poprithms
