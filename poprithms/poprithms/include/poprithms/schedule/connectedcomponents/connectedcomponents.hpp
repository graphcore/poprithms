// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_CONNECTEDCOMPONENTS_CONNECTEDCOMPONENTS_HPP
#define POPRITHMS_SCHEDULE_CONNECTEDCOMPONENTS_CONNECTEDCOMPONENTS_HPP

#include <tuple>
#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace schedule {
namespace connectedcomponents {

/**
 * Each disjoint subgraph has a distinct ComponentId. Every node in the graph
 * has exactly 1 ComponentId, corresponding to the subgraph to which it
 * belongs.
 * */
using ComponentId = util::TypedInteger<'C', uint32_t>;

/**
 * Within a subgraph, each node has a 'local' id.
 * */
using LocalId = util::TypedInteger<'L', uint32_t>;

/**
 * The edges between nodes in a graph. Note that the graph may contain cycles.
 * */
template <typename T> using Edges = std::vector<std::vector<T>>;

/**
 * A partitioning of a graph into connected components.
 * */
class ConnectedComponents {

public:
  /**
   * Construct an object from the full graph's edges. If b is in edges[a],
   * this means that there is an edge between a and b.
   * */
  explicit ConnectedComponents(const Edges<int64_t> &edges);
  explicit ConnectedComponents(const Edges<uint64_t> &edges);

  /**
   * The total number of disjoint subgraphs.
   * */
  uint64_t nComponents() const { return toGlobal.size(); }

  /**
   * The number of nodes in subgraph #id.
   * */
  uint64_t nNodes(ComponentId id) const {
    return toGlobal.at(id.get()).size();
  }

  /**
   * The edges in the subgraph #c. Note that nodes have 2 ids, a global id,
   * which identifies them in the main graph, and a local id, which identifies
   * them in the subgraph which contains them. The ids in this subgraph are
   * the local ones.
   * */
  const Edges<uint64_t> &component(ComponentId c) const {
    return components.at(c.get());
  }

  /**
   * The subgraph to which the node #mainId in the main graph belongs.
   * */
  ComponentId componentId(uint64_t mainId) const {
    return std::get<0>(toLocal[mainId]);
  }

  /**
   * The id within the subgraph of the main graph node, #mainId.
   * */
  LocalId localId(uint64_t mainId) const {
    return std::get<1>(toLocal[mainId]);
  }

  /**
   * The id in the main graph of the the node in the subgraph #c with local
   * id, #l.
   * */
  uint64_t globalId(ComponentId c, LocalId l) const {
    return toGlobal[c.get()][l.get()];
  }

  void append(std::ostream &) const;

private:
  // A node in the main (possibly disconnected) graph can be identified by
  // 1) its 'global' id in the main graph, or
  // 2) its 'local' id in its connected component, and the id of the compoent
  // itself.
  //

  // mapping from (1) to (2)
  std::vector<std::tuple<ComponentId, LocalId>> toLocal;

  // mapping from (2) to (1). toGlobal[componentId][localId] -> globalId.
  std::vector<std::vector<uint64_t>> toGlobal;

  // local edges
  std::vector<Edges<uint64_t>> components;

  // The bool argument is a dummy variable to distinguish this constructor
  // from the public ones
  template <typename T> ConnectedComponents(const Edges<T> &edges, bool);
};

std::ostream &operator<<(std::ostream &, const ConnectedComponents &);

} // namespace connectedcomponents
} // namespace schedule
} // namespace poprithms

#endif
