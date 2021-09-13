// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_FIFOSTACK_HPP
#define POPRITHMS_SCHEDULE_VANILLA_FIFOSTACK_HPP

#include <sstream>

#include <schedule/vanilla/basestackwithmanypriorities.hpp>
#include <schedule/vanilla/kahn.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {
namespace fifo {

/**
 * Filo tie-breaking is possibly the fastest way to do tie-breaking with Kahn,
 * as no stack is required. An index is kept in the schedule to track which
 * node to process (message to downstream ops that it's scheduled) next.
 * */
template <typename TNode> class SchedulerWithoutPriorities {
public:
  static std::vector<TNode> kahn(const Edges<TNode> &fwdEdges) {

    // The total number of nodes in the graph.
    const auto N = fwdEdges.size();

    // The number of nodes which must be scheduled before a node is scheduled.
    auto nOutstandingDeps = getOutstandingCount(fwdEdges);

    // The index of the current node from which to signal that it is
    // scheduled.
    uint64_t current{0};

    std::vector<TNode> schedule;
    schedule.reserve(N);

    // Insert all the nodes which have no input dependencies into the
    // schedule.
    for (uint64_t i = 0; i < N; ++i) {
      if (nOutstandingDeps[i] == 0) {
        schedule.push_back(i);
      }
    }

    // process the node at index 'current' and increment 'current'. This is
    // the first-in-first-out approach.
    while (current != schedule.size()) {
      for (const auto to : fwdEdges[schedule[current]]) {
        --nOutstandingDeps[to];
        if (nOutstandingDeps[to] == 0) {
          schedule.push_back(to);
        }
      }
      current += 1;
    }

    return schedule;
  }
};

/**
 * Fifo scheduling with priorities requires a stack, and so looks very similar
 * to filo scheduling.
 * */
template <typename TNode, typename TPriority>
class StackWithManyPriorities
    : public BaseFifoStackWithManyPriorities<TNode, TPriority> {
public:
  StackWithManyPriorities(uint64_t n, const Priorities<TNode, TPriority> &ps)
      : BaseFifoStackWithManyPriorities<TNode, TPriority>(n, ps) {}

  TNode pop() { return this->basePop(); }

  void push(TNode t) {
    this->ready.push_back(
        {t, this->sparsePriorities[static_cast<uint64_t>(t)]});
  }
};

template <typename TNode, typename TPriority>
class SchedulerWithManyPriorities {
public:
  static std::vector<TNode>
  kahn(const Edges<TNode> &fwdEdges,
       const Priorities<TNode, TPriority> &priorities) {
    auto a_ = StackWithManyPriorities<TNode, TPriority>(fwdEdges.size(),
                                                        priorities);
    return stackBasedKahn<TNode>(fwdEdges, a_);
  }
};

template <typename TNode, typename TPriority>
static std::vector<TNode> kahn(const Edges<TNode> &fwdEdges,
                               const Priorities<TNode, TPriority> &priorities,
                               const Links<TNode> &links,
                               ErrorIfCycle eic,
                               VerifyEdges ve) {
  return delegate<TNode,
                  TPriority,
                  SchedulerWithoutPriorities<TNode>,
                  SchedulerWithManyPriorities<TNode, TPriority>>(
      fwdEdges, eic, ve, priorities, links, [](const auto &) {
        return std::tuple<>();
      });
}

} // namespace fifo
} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
