// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <limits>
#include <numeric>
#include <schedule/scc/error.hpp>
#include <set>
#include <sstream>
#include <unordered_map>

#include <poprithms/schedule/dfs/dfs.hpp>
#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/typedinteger.hpp>
#include <poprithms/util/typedvector.hpp>

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
        oss << "Failure in confirmValidEdges. "
            << "Invalid edge in Graph with " << N << " nodes (" << from
            << "->" << d << ')';
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

std::vector<std::vector<uint64_t>> getCycles(const SCCs &sccs,
                                             const FwdEdges &fwdEdges) {

  // this function performs a depth first search until it returns to the
  // starting point of the search. If during the depth first search the
  // algorithm returns to a node which it's seen before (other than the node
  // which it started to cycle search from), we abandon that search path.
  //
  // This vector records all nodes which have been visited, in all components.
  std::vector<bool> visited(fwdEdges.size(), false);

  // Get the shortest cycle starting at start (if one exists, otherwise return
  // the empty cycle).
  auto getNextCycle = [&visited, &fwdEdges](uint64_t start) {
    // sources[x] contains predecesor to x on a shortest path from start to x.
    std::unordered_map<uint64_t, uint64_t> sources;

    // Depth first search data structures. We process all the nodes in
    // toProcess0, collecting any new unexplored nodes in toProcess1 for the
    // next round.
    std::vector<uint64_t> toProcess0{start};
    std::vector<uint64_t> toProcess1{};

    visited[start] = true;

    while (!toProcess0.empty()) {
      for (auto current : toProcess0) {
        for (auto nxt : fwdEdges.at(current)) {

          // if we've found a cycle (start->...->start) we reconstruct the
          // path and return it. This path is (one of) the shortest cycle(s)
          // starting from 'start'
          if (nxt == start) {

            // first construct the path in reverse order:
            std::vector<uint64_t> stack_{start};
            while (current != start) {
              stack_.push_back(current);
              current = sources.at(current);
            }
            stack_.push_back(start);

            // and then reverse it, so that it follows the forward edges in
            // #fwdEdges.
            std::reverse(stack_.begin(), stack_.end());
            return stack_;
          }

          if (!visited[nxt]) {
            sources.insert({nxt, current});
            toProcess1.push_back(nxt);
            visited[nxt] = true;
          }
        }
      }
      std::swap(toProcess0, toProcess1);
      toProcess1.clear();
    }

    // In the case of singleton component without a edge to itself, there is
    // not cycle.
    return std::vector<uint64_t>{};
  };

  std::vector<std::vector<uint64_t>> cycles;
  cycles.reserve(sccs.size());
  for (auto iter = sccs.crbegin(); iter != sccs.crend(); ++iter) {
    uint64_t start = iter->at(0);
    cycles.push_back(getNextCycle(start));
  }

  std::reverse(cycles.begin(), cycles.end());

  return cycles;
}

// Example:
//
// >  Strongly Connected Component #2:
// >    Op (from)  from  to
// >    ---------- ----- ---
// >    30_gamma   2     (3)
// >    31_delta   3     (4)
// >    32_epsilon 4     (2)
//
std::string getSummary(const FwdEdges &edges,
                       const std::vector<std::string> &dbs,
                       IncludeSingletons singletons) {
  if (dbs.size() != edges.size()) {
    std::ostringstream oss;
    oss << "Bad input to getSummary. "
        << "FwdEdges is of size " << edges.size()
        << ", and dbs (the debug strings) is of size " << dbs.size()
        << ". They must be equal. ";
    throw error(oss.str());
  }

  const auto components = getStronglyConnectedComponents(edges);
  const auto cycles     = getCycles(components, edges);

  std::ostringstream oss;

  using util::StringColumn;

  for (uint64_t ci = 0; ci < components.size(); ++ci) {
    if (singletons == IncludeSingletons::Yes || components[ci].size() > 1) {
      oss << "\n\nStrongly Connected Component #" << ci << ": " << '\n';

      // We map all nodes in components to a contiguous set starting from 0.
      const auto toLocal = [&edges, &components, &ci]() {
        std::set<uint64_t> allNodes{components[ci].cbegin(),
                                    components[ci].cend()};
        for (auto n : components[ci]) {
          for (auto e : edges[n]) {
            allNodes.insert(e);
          }
        }

        uint64_t l{0};
        std::unordered_map<uint64_t, uint64_t> toLocalMap;
        for (auto n : allNodes) {
          toLocalMap.insert({n, l});
          ++l;
        }
        return toLocalMap;
      }();

      auto getLocal = [&toLocal](const std::vector<uint64_t> &globals) {
        std::vector<uint64_t> locals;
        locals.reserve(globals.size());
        for (auto g : globals) {
          locals.push_back(toLocal.at(g));
        }
        return locals;
      };

      std::vector<std::string> componentDbs;
      std::vector<std::vector<uint64_t>> componentTo;
      for (auto opAddress : components[ci]) {
        componentDbs.push_back(dbs[opAddress]);
        componentTo.push_back(getLocal(edges[opAddress]));
      }
      const auto componentFrom = getLocal(components[ci]);
      const auto sp            = getLocal(cycles.at(ci));

      oss << util::alignedColumns(
          {{"Op (debug name)", componentDbs},
           {"Op (local id)", StringColumn::entriesFromInts(componentFrom)},
           {"Edge ends (local ids)",
            StringColumn::entriesFromVectors(componentTo)}});

      if (!sp.empty()) {
        oss << "\nOne cycle (out of potentially many) in this Strongly "
               "Connected Component:  ";
        oss << '(' << sp[0];
        for (uint64_t i = 1; i < sp.size(); ++i) {
          oss << "->" << sp[i];
        }
        oss << ')';
      }
    }
  }
  return oss.str();
}

template <typename F, typename T>
std::vector<std::vector<T>> translate(const std::vector<std::vector<F>> &f) {
  std::vector<std::vector<T>> t;
  t.reserve(f.size());
  for (const auto &v : f) {
    t.push_back({v.cbegin(), v.cend()});
  }
  return t;
}

std::string getSummary_i64(const FwdEdges_i64 &edges_i64,
                           const std::vector<std::string> &dbs,
                           IncludeSingletons singletons) {
  return getSummary(translate<int64_t, uint64_t>(edges_i64), dbs, singletons);
}

} // namespace scc
} // namespace schedule
} // namespace poprithms
