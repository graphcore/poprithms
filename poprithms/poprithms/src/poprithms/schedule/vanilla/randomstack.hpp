// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_RANDOMSTACK_HPP
#define POPRITHMS_SCHEDULE_VANILLA_RANDOMSTACK_HPP

#include <random>

#include <schedule/vanilla/basestackwithmanypriorities.hpp>
#include <schedule/vanilla/kahn.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {
namespace random {

// random tie-breaker.

template <typename TNode> class StackWithoutPriorities {
public:
  StackWithoutPriorities(uint64_t /* nNodes */, uint32_t seed) : rng(seed) {}
  TNode pop() {
    // get a node at random from the stack. If the stack isn't already in a
    // random we shuffle it:
    if (!isRandom) {
      std::shuffle(ready.begin(), ready.end(), rng);
      isRandom = true;
    }
    // now that we know the stack is in a random order, we take the node at
    // the back.
    auto x = ready.back();
    ready.pop_back();
    return x;
  }
  void push(TNode t) {
    ready.push_back(t);
    // The final node in the stack is now the most recently added node. The
    // stack is therefore not random:
    isRandom = false;
  }
  bool empty() const { return ready.empty(); }

private:
  std::vector<TNode> ready;

  // Is the order of the nodes in the stack random?
  bool isRandom{false};
  std::mt19937 rng;
};

template <typename TNode> class SchedulerWithoutPriorities {
public:
  static std::vector<TNode> kahn(const Edges<TNode> &fwdEdges,
                                 uint32_t seed) {
    auto a_ = StackWithoutPriorities<TNode>(fwdEdges.size(), seed);
    return stackBasedKahn<TNode>(fwdEdges, a_);
  }
};

template <typename TNode, typename TPriority>
class StackWithManyPriorities
    : public BaseFiloStackWithManyPriorities<TNode, TPriority> {

  // See StackWithoutPriorities for a description of the logic around
  // 'isRandom'
public:
  StackWithManyPriorities(uint64_t n,
                          const Priorities<TNode, TPriority> &priorities,
                          uint32_t seed)
      : BaseFiloStackWithManyPriorities<TNode, TPriority>(n, priorities),
        rng(seed) {}

  TNode pop() {
    if (!isRandom) {
      std::shuffle(this->ready.begin(), this->ready.end(), rng);
      isRandom = true;
    }
    return this->basePop();
  }

  void push(TNode t) {
    isRandom = false;
    this->ready.push_back(
        {t, this->sparsePriorities[static_cast<uint64_t>(t)]});
  }

private:
  std::mt19937 rng;
  bool isRandom{true};
};

template <typename TNode, typename TPriority>
class SchedulerWithManyPriorities {
public:
  static std::vector<TNode>
  kahn(const Edges<TNode> &fwdEdges,
       const Priorities<TNode, TPriority> &priorities,
       uint64_t seed) {
    auto a_ = StackWithManyPriorities(fwdEdges.size(), priorities, seed);
    return stackBasedKahn<TNode>(fwdEdges, a_);
  }
};

template <typename TNode, typename TPriority>
static std::vector<TNode> kahn(const Edges<TNode> &fwdEdges,
                               const Priorities<TNode, TPriority> &priorities,
                               const Links<TNode> &links,
                               uint32_t seed,
                               ErrorIfCycle eic,
                               VerifyEdges ve) {

  auto tg = [seed](const auto &) { return std::tuple<uint32_t>(seed); };

  return delegate<TNode,
                  TPriority,
                  SchedulerWithoutPriorities<TNode>,
                  SchedulerWithManyPriorities<TNode, TPriority>>(
      fwdEdges, eic, ve, priorities, links, tg, seed);
}

} // namespace random
} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
