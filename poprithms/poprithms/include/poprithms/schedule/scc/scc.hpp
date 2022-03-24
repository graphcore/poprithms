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

enum class IncludeCyclelessComponents { No = 0, Yes };

/// \deprecated {on 10 March 2022. Please use IncludeCyclelessComponents.}
using IncludeSingletons = IncludeCyclelessComponents;

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
                       const std::vector<std::string> &debugStrings,
                       IncludeCyclelessComponents sings);

/**
 * For a customized summary which includes edge information (is an edge a data
 * dependency, a control dependency, or something else?), complete the
 * following abstract base class and use it in the method getSummary which
 * follows.
 * */
class NodeInfoGetter {
public:
  /**
   * A debug string for the n'th node.
   * */
  virtual std::string nodeString(uint64_t n) const = 0;

  /**
   * \return true if the summary should include edge information.
   * */
  virtual bool providesEdgeStrings() const = 0;

  /**
   * A debug string for the edge from node #f to node #t. This method is only
   * used if #providesEdgeStrings returns true.
   * */
  virtual std::string edgeString(uint64_t f, uint64_t t) const = 0;
};
std::string getSummary(const FwdEdges &edges,
                       const NodeInfoGetter &,
                       IncludeSingletons sings);

/**
 * Same as getSummary, but the node ids are int64_t.
 *
 * \sa getSummary
 * */
using FwdEdges_i64 = std::vector<std::vector<int64_t>>;
std::string getSummary_i64(const FwdEdges_i64 &,
                           const std::vector<std::string> &debugStrings,
                           IncludeCyclelessComponents);

/**
 * Return one cycle from every strongly connected component.
 *
 * \param SCCs The strongly connected components of the graph with forward
 *             edges #edges. SCCs must be in topological order.
 *
 * \param edges The edges of the directed graph with strongly connected
 *              components #SCCs.
 *
 * The only case where a strongly connected component does not have a cycle,
 * is when it is a singleton with no self-edge. For these, the empty cycle is
 * returned.
 *
 * Example:
 *
 *   a->b->c--->e--->f---+
 *   |     |         |   |
 *   +--<--+         +-<-+
 *   |     |
 *   +<-d<-+
 *
 * has connected components in topological order ((a,b,c,d), (e), (f)).
 * The cycles returned might be:
 *   (a,b,c), (), (f)
 *
 * The complexity of this algorithm is O(E), where E is the number of edges in
 * #edges.
 *
 * Note that there is no guarantee as to which cycle in the component
 * (a,b,c,d) will be returned. It might not be shortest, and it might not
 * traverse all nodes in the component. The current implementation will return
 * the shortest cycle which includes the first node (a), but this might change
 * in the future.
 * */
std::vector<std::vector<uint64_t>> getCycles(const SCCs &sccs,
                                             const FwdEdges &);

} // namespace scc
} // namespace schedule
} // namespace poprithms

#endif
