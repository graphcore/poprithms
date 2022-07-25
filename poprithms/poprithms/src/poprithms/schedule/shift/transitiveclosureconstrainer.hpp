// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_TRANSITIVECLOSURECONSTRAINER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_TRANSITIVECLOSURECONSTRAINER_HPP

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * Transformations to a graph which insert constraints and links between Ops.
 * The transformations are all designed to preserve min_(schedules)(mean
 * liveness) while reducing |schedules|. That is, they reduce the search space
 * without removing the global minimum.
 *
 * See the class TransitiveClosureOptimizations for more information on what
 * each of the transformations does.
 * */
class TransitiveClosureConstrainer {
public:
  /**
   * \param g The graph to transform
   *
   * \param tc The transitive closure of all edges in the graph #g.
   *
   * \param lows Lower bounds on the change in liveness that scheduling each
   *             Op might have. Specifically, these bounds must be valid for
   *             all possible graph schedules.
   *
   * \param upps Upper bounds on the change in liveness that scheduling each
   *             Op might have. Rhese bounds must be valid for all possible
   *             graph schedules.
   *
   * To apply the transform 'foo' to a graph #g:
   *
   * <code>
   *   auto changed = TransitiveClosureConstrainer(g, tc, lows, upps).foo();
   * </code>
   *
   * The returned boolean specifies if #g changed.
   *
   * */
  TransitiveClosureConstrainer(Graph &g,
                               const transitiveclosure::TransitiveClosure &tc,
                               const std::vector<AllocWeight> &lows,
                               const std::vector<AllocWeight> &upps)
      : graph(g), transitiveClosure(tc), lowerBoundChange(lows),
        upperBoundChange(upps) {}

  bool constrainParallelChains() const;

  bool slideLinks() const;

  bool linkCloseTightPairs() const;

  bool linkTightDrops() const;

  bool constrainWeightSeparatedGroups() const;

private:
  Graph &graph;
  const transitiveclosure::TransitiveClosure &transitiveClosure;
  const std::vector<AllocWeight> &lowerBoundChange;
  const std::vector<AllocWeight> &upperBoundChange;

  // A class for recording which ops are visited during depth-first search,
  // used to accelerate the inner loop of one of the optimization passes.
  class DfsVisitRecords;

  // Internal method used by constrainWeightSeparatedGroups.
  void processWeightSeparatedIdenticalIns(
      const std::vector<OpAddress> &opsWithIdenticalIns,
      std::vector<std::array<OpAddress, 2>> &cons,
      DfsVisitRecords &) const;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
