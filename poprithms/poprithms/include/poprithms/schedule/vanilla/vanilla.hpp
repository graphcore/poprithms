// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_VANILLA_HPP
#define POPRITHMS_SCHEDULE_VANILLA_VANILLA_HPP

#include <array>
#include <tuple>
#include <vector>

#include <poprithms/schedule/vanilla/types.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

/**
 * Schedulable nodes with high priority will be scheduled before schedulable
 * nodes with low priority. Nodes with no priority in this vector will have
 * priority of zero.
 * */

template <typename TNode, typename TPriority>
using Priorities = std::vector<std::tuple<TNode, TPriority>>;

/**
 * Nodes which are linked will always be scheduled contiguously.
 * */
template <typename TNode> using Links = std::vector<std::array<TNode, 2>>;

/**
 * Example. If
 *
 * fwdEdges[0] = {1,2}
 * fwdEdges[1] = {3}
 * fwdEdges[2] = {3}
 * fwdEdges[3] = {}
 *
 * This denotes the dependency Graph,
 *
 *
 *  +--> 1 --+
 *  |        |
 *  |        v
 *  |
 *  0        3
 *  |
 *  |        ^
 *  |        |
 *  +--> 2 --+
 *
 *  fwdEdges may contain repeated edges.
 *
 * These two 'give-me-any-schedule' methods, currently the tie-breaker used is
 * FILO (first-in, last-out) but this might change in the future.
 *
 * */
std::vector<uint64_t>
getSchedule_u64(const Edges<uint64_t> &fwdEdges, ErrorIfCycle, VerifyEdges);

std::vector<int64_t>
getSchedule_i64(const Edges<int64_t> &fwdEdges, ErrorIfCycle, VerifyEdges);

/**
 * Definition of 'schedulable': any node which has had all of its input
 * dependencies satisfied, and so is ready to be scheduled, but has not yet
 * been scheduled.
 *
 * This class implements Kahn's algorithm, with various tie-breakers for
 * deciding which schedulable node is scheduled at any moment.
 *
 * Kahn's algorithm:
 * https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
 *
 * In all scheduling methods, the 'executive' decision of which node to
 * schedule is controlled by Priorities. A Priority is a <node id, priority
 * value> pair. Nodes which do not have a Priority get the default value,
 * zero.
 *
 * Schedulable node(s) which do not have the highest priority value of all
 * schedulable nodes are not considered for scheduling. The 'secondary'
 * decision, which is used when there's a tie of priorities, is method
 * specific:
 *
 * filo   : the most recently found schedulable node will be scheduled
 * fifo   : the least recently found schedulable node will be scheduled
 * random : a random schedulabe node will be scheduled.
 *
 * */
template <typename TNode, typename TPriority> class Scheduler {

public:
  /**
   * First-in-last-out tie-breaking
   * */
  static std::vector<TNode> filo(const Edges<TNode> &fwdEdges,
                                 const Priorities<TNode, TPriority> &,
                                 const Links<TNode> &,
                                 ErrorIfCycle,
                                 VerifyEdges);

  /**
   * First-in-first-out tie-breaking
   * */
  static std::vector<TNode> fifo(const Edges<TNode> &fwdEdges,
                                 const Priorities<TNode, TPriority> &,
                                 const Links<TNode> &,
                                 ErrorIfCycle,
                                 VerifyEdges);

  /**
   * Random tie-breaking, randomness determined by the #seed value.
   * */
  static std::vector<TNode> random(const Edges<TNode> &fwdEdges,
                                   const Priorities<TNode, TPriority> &,
                                   const Links<TNode> &,
                                   uint32_t seed,
                                   ErrorIfCycle,
                                   VerifyEdges);
};

extern template class Scheduler<int64_t, double>;
extern template class Scheduler<uint64_t, double>;

/**
 * A kahn tie-breaker which chooses the Op which results in the largest
 * immediate liveness reduction at every step.
 * */
template <typename TNode, typename TPriority, typename TAllocSize>
class GreedyScheduler {
public:
  static std::vector<TNode>
  kahn(const Edges<TNode> &fwdEdges,
       const Priorities<TNode, TPriority> &priorities,
       const Links<TNode> &links,
       const std::vector<TAllocSize> &sizes,
       const Edges<TNode> &allocsToNodes,
       ErrorIfCycle eic,
       VerifyEdges ve);
};

extern template class GreedyScheduler<int64_t, double, int>;
extern template class GreedyScheduler<uint64_t, double, int>;

template <typename TNode> class Query {

public:
  /**
   * Return true if the graph defined by the edges contains no cycles.
   * */
  static bool isSchedulable(const Edges<TNode> &edges, VerifyEdges);

  /**
   * Return true if the graph defined by the edges and links is schedulable.
   * */
  static bool
  isSchedulable(const Edges<TNode> &edges, const Links<TNode> &, VerifyEdges);

  /**
   * Return true if there is exactly 1 way to schedule the graph with forward
   * edges #fwdEdges. If there are zero (due to a cycle) or multiple ways to
   * schedule the graph, then false is returned.
   * */
  static bool hasUniqueSchedule(const Edges<TNode> &fwdEdges, VerifyEdges);
};

extern template class Query<int64_t>;
extern template class Query<uint64_t>;

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
