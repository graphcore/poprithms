// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <ostream>
#include <sstream>

#include <poprithms/schedule/connectedcomponents/connectedcomponents.hpp>
#include <poprithms/schedule/connectedcomponents/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace connectedcomponents {

template <typename T>
ConnectedComponents::ConnectedComponents(const Edges<T> &edges, bool) {

  const auto N = edges.size();
  toLocal.resize(N);

  // for every edge a->b in edges, the object bidirectional (below) will
  // contain a-> and b->a.
  auto bidirectional = edges;
  for (uint64_t i = 0; i < N; ++i) {
    for (auto j : edges[i]) {
      if (static_cast<uint64_t>(j) >= N) {
        std::ostringstream oss;
        oss << "Invalid end of edge, '" << j << "', in edge map of size " << N
            << '.';
        throw schedule::connectedcomponents::error(oss.str());
      }
      bidirectional[j].push_back(i);
    }
  }

  std::vector<bool> visited(N, false);
  std::vector<uint64_t> toProcess;

  auto enqueue = [&visited, &toProcess](uint64_t i) {
    visited[i] = true;
    toProcess.push_back(i);
  };
  for (uint64_t i = 0ull; i < N; ++i) {
    if (!visited[i]) {
      // Create a new componentgraph:
      const auto componentgraph = toGlobal.size();
      toGlobal.push_back({});

      // starting at node i, do depth first traversal, adding nodes as you
      // visit them.
      enqueue(i);
      while (!toProcess.empty()) {
        const auto nxt = toProcess.back();
        toProcess.pop_back();
        toLocal[nxt] = {componentgraph, toGlobal.back().size()};
        toGlobal.back().push_back(nxt);
        for (auto end_ : bidirectional[nxt]) {
          if (!visited[end_]) {
            enqueue(end_);
          }
        }
      }
    }
  }

  // set components. These are the subgraphs, expressed using local IDs.
  for (uint64_t c = 0; c < nComponents(); ++c) {
    Edges<uint64_t> e;
    e.reserve(nNodes(c));
    for (uint64_t i = 0; i < nNodes(c); ++i) {
      const auto globalId = toGlobal[c][i];
      e.push_back({});
      e.back().reserve(edges[globalId].size());
      for (auto end_ : edges[globalId]) {
        e.back().push_back(localId(end_).get());
      }
    }
    components.push_back(e);
  }
}

template <typename T>
const Edges<T> &assertAllPositive(const Edges<T> &edges) {
  for (const auto &vs : edges) {
    for (auto v : vs) {
      if (v < 0) {
        throw error("All edge destinations must be positive in "
                    "ConnectedComponents construction");
      }
    }
  }
  return edges;
}

void ConnectedComponents::append(std::ostream &ost) const {
  for (uint64_t i = 0ull; i < nComponents(); ++i) {
    ost << "In component " << i << " : ";
    auto inComponent = toGlobal.at(i);
    std::sort(inComponent.begin(), inComponent.end());
    util::append(ost, toGlobal.at(i));
    if (i + 1 < nComponents()) {
      ost << '\n';
    }
  }
}

std::ostream &operator<<(std::ostream &ost, const ConnectedComponents &ccs) {
  ccs.append(ost);
  return ost;
}

ConnectedComponents::ConnectedComponents(const Edges<int64_t> &edges)
    : ConnectedComponents(assertAllPositive(edges), false) {}

ConnectedComponents::ConnectedComponents(const Edges<uint64_t> &edges)
    : ConnectedComponents(edges, false) {}

} // namespace connectedcomponents
} // namespace schedule
} // namespace poprithms
