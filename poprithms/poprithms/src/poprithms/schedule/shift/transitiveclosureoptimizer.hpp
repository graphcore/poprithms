// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_TRANSITIVECLOSUREOPTIMIZER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_TRANSITIVECLOSUREOPTIMIZER_HPP

#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class TransitiveClosureOptimizer {

  using TimeLogger = poprithms::logging::SwitchingTimePartitionLogger;

public:
  /**
   * Apply the set of transitive closure optimizations in #tcos to the Graph
   * #g.
   * */
  static void
  apply(const TransitiveClosureOptimizations &tcos, Graph &g, TimeLogger &);

private:
  TransitiveClosureOptimizer(const TransitiveClosureOptimizations &tcos,
                             Graph &g,
                             TimeLogger &tl)
      : graph(g), timeLogger_(tl) {
    applyTransitiveClosureOptimizations(tcos);
  }

  // Insert constraints and links which can be proven to satisfy at least one
  // globally minimizing schedule. These constraints accelerate the shifting
  // algorithm by reducing its search space.
  void
  applyTransitiveClosureOptimizations(const TransitiveClosureOptimizations &);

  // create the transitive closure from scratch.
  void initializeTransitiveClosure();

  // update the transitive closure with all of the graph edges. The edges must
  // be a superset of the edges used to update/create the transitive closure
  // previously.
  void reinitializeTransitiveClosure();

  void removeRedundantEdges();

  // Incrementally update the TransitiveClosure of this Graph. Note that the
  // TransitiveClosure can be initialized with
  // updateTransitiveClosure(getForwardEdges()), but it is less efficient than
  // calling initializeTransitiveClosure().
  void
  updateTransitiveClosure(const std::vector<std::vector<OpAddress>> &nEdges);

  void finalizeTransitiveClosure();

private:
  transitiveclosure::TransitiveClosure transitiveClosure{{}};
  // The lowest change in liveness across all schedules, for each Op
  std::vector<AllocWeight> lowerBoundChange;
  // The highest change in liveness across all schedules, for each Op
  std::vector<AllocWeight> upperBoundChange;

  TimeLogger &timeLogger() { return timeLogger_; }

  Graph &graph;
  Graph &getGraph() { return graph; }
  uint64_t nOps() const { return graph.nOps(); }
  TimeLogger &timeLogger_;
};
} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
