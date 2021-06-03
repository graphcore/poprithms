// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_TRANSITIVECLOSURE_PARTITIONEDTRANSITIVECLOSURE_HPP
#define POPRITHMS_SCHEDULE_TRANSITIVECLOSURE_PARTITIONEDTRANSITIVECLOSURE_HPP

#include <algorithm>
#include <numeric>

#include <poprithms/schedule/connectedcomponents/connectedcomponents.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

/**
 * The (CPU) memory used by a TransitiveClosure with N ops is (exactly)
 * 2*N*ceil(N/BitSetSize)*BitSetSize bits. This is quadratic in N, but with a
 * small constant in the complexity because of an efficient std::bitset
 * implementation. Nonetheless, 'quadratic in N' becomes prohibitive for very
 * large graphs.
 *
 * With a PartitionedTransitiveClosure, the memory footprint scales
 * quadratically in the size of the \b largest connected component. Thus for
 * graphs which are composed of multiple disconnected sub-graphs, there can be
 * a significant memory saving.
 * */
class PartitionedTransitiveClosure {
public:
  /**
   * Construct a transitive closure from the forward edges of a graph. The
   * forward edges might consist of multiple disjoint subgraphs.
   * */
  PartitionedTransitiveClosure(const Edges &forwardEdges);

  /**
   * Return true if there is a constraint (implicit or explicit) that #from
   * must be sheduled before #to. In other words, return true if there exist
   * no schedules with #to before #from. This query is O(1).
   * */
  bool constrained(OpId from, OpId to) const;

  /**
   * Return true if there is no constraint a->b and no constraint b->a.
   * */
  bool unconstrainedInBothDirections(OpId a, OpId b) const {
    return !constrained(a, b) && !constrained(b, a);
  }

  /**
   * The total number of connected components in the graph which this
   * PartitionedTransitiveClosure describes.
   * */
  uint64_t nComponents() const { return ccs.nComponents(); }

  /**
   * the total size of all bitmaps used by this object
   * */
  uint64_t nBits() const;

  // TODO(T40029)
  // This class should be made closer to feature complete, exposing the same
  // functionality as TransitiveClosure. The one tricky method to implement
  // will be update, which will involve merging of TransitiveClosures when a
  // new edge connects previously disjoint components.

private:
  // Each connected component has its own TransitiveClosure:
  std::vector<TransitiveClosure> transitiveClosures;
  schedule::connectedcomponents::ConnectedComponents ccs;
};

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms

#endif
