// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_FILOSTACK_HPP
#define POPRITHMS_SCHEDULE_VANILLA_FILOSTACK_HPP

#include <schedule/vanilla/basestackwithmanypriorities.hpp>
#include <schedule/vanilla/kahn.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {
namespace filo {

// First-in-last-out tie-breaking.

template <typename TNode> class StackWithoutPriorities {
public:
  StackWithoutPriorities(uint64_t) {}

  // return the node which was most recently added to the stack:
  // last-in-first-out ( == first-in-last-out).
  TNode pop() {
    auto x = ready.back();
    ready.pop_back();
    return x;
  }
  void push(TNode t) { ready.push_back(t); }
  bool empty() const { return ready.empty(); }

private:
  std::vector<TNode> ready;
};

template <typename TNode> class SchedulerWithoutPriorities {
public:
  static std::vector<TNode> kahn(const Edges<TNode> &fwdEdges) {
    auto schedulerStack = StackWithoutPriorities<TNode>(fwdEdges.size());
    return stackBasedKahn<TNode>(fwdEdges, schedulerStack);
  }
};

template <typename TNode, typename TPriority>
class StackWithManyPriorities
    : public BaseFiloStackWithManyPriorities<TNode, TPriority> {
public:
  StackWithManyPriorities(uint64_t n, const Priorities<TNode, TPriority> &ps)
      : BaseFiloStackWithManyPriorities<TNode, TPriority>(n, ps) {}

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
    auto schedulerStack = StackWithManyPriorities<TNode, TPriority>(
        fwdEdges.size(), priorities);
    return stackBasedKahn<TNode>(fwdEdges, schedulerStack);
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
        // there are no arguments specific to filo scheduling, unlike the
        // random scheduler for example, which has a random seed. We therefore
        // return the empty tuple for this method, which converts tie-breaker
        // specific arguments when compressing graphs with links.
        return std::tuple<>();
      });
}

} // namespace filo
} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
