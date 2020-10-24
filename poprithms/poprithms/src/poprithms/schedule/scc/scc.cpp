// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poprithms/util/typedvector.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

#include <poprithms/schedule/dfs/dfs.hpp>
#include <poprithms/schedule/scc/error.hpp>
#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace schedule {
namespace scc {

namespace {

using Component  = util::TypedInteger<'C', uint64_t>;
using Components = std::vector<Component>;
const Component Undefined{std::numeric_limits<uint64_t>::max()};

// Replace all a->b edges with b->a edges, i.e. reverse a directed graph.
using BwdEdges = FwdEdges;
BwdEdges getReverseEdges(const FwdEdges &fwdEdges) {
  const auto N = fwdEdges.size();
  BwdEdges bwdEdges(N);
  for (uint64_t from = 0; from < N; ++from) {
    for (auto to : fwdEdges[from]) {
      bwdEdges[to].push_back(from);
    }
  }
  return bwdEdges;
}

/**
 *
 * Starting at the node \a start, traverse through all nodes which do not have
 * a defined Component, and set Component to \a c.
 *
 * */
void depthFirstComponentFill(uint64_t start,
                             Component c,
                             Components &components,
                             const FwdEdges &fwdEdges) {

  if (components[start] != Undefined) {
    return;
  }

  components[start] = c;
  std::vector<uint64_t> ready{start};

  while (!ready.empty()) {
    auto nxt = ready.back();
    ready.pop_back();
    for (auto to : fwdEdges[nxt]) {
      if (components[to] == Undefined) {
        components[to] = c;
        ready.push_back(to);
      }
    }
  }
}

void confirmValidEdges(const FwdEdges &edges) {
  const auto N = edges.size();
  for (uint64_t from = 0; from < N; ++from) {
    const auto &destinations = edges[from];
    for (auto d : destinations) {
      if (d >= N) {
        std::ostringstream oss;
        oss << "Invalid edge in Graph with " << N << "nodes (" << from << "->"
            << d << ')';
        throw error(oss.str());
      }
    }
  }
}

} // namespace

SCCs getStronglyConnectedComponents(const FwdEdges &edges) {

  confirmValidEdges(edges);

  const auto N = edges.size();

  const auto revEdges = getReverseEdges(edges);

  // highest post-order is in the final component of the forward graph.
  //
  // See the book "Algorithms" by
  // S. Dasgupta, C. H. Papadimitriou, and U. V. Vaziran
  // Property 3, page 93.
  //
  const auto postOrder = dfs::postOrder(revEdges);

  std::vector<uint64_t> visitOrder = postOrder;
  std::reverse(visitOrder.begin(), visitOrder.end());
  std::vector<Component> components(N, Undefined);
  uint64_t component = 0;
  uint64_t start     = 0;
  while (start < N) {
    const auto node = visitOrder[start];
    if (components[node] == Undefined) {
      depthFirstComponentFill(node, component, components, edges);
      ++component;
    }
    ++start;
  }

  // Nodes in the final connected component of the forward graph, have
  // component=0. We invert the components in constructing the final SCC.
  const auto nComps = component;
  SCCs sccs(nComps);
  for (uint64_t i = 0; i < N; ++i) {
    const auto dagComponent = nComps - components[i].get() - 1;
    sccs[dagComponent].push_back(i);
  }
  return sccs;
}
} // namespace scc
} // namespace schedule
} // namespace poprithms
