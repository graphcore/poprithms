// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_BASESTACKWITHMANYPRIORITIES_HPP
#define POPRITHMS_SCHEDULE_VANILLA_BASESTACKWITHMANYPRIORITIES_HPP

#include <tuple>
#include <vector>

#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

/**
 * This stack is implemented for the case where the majority of nodes have
 * priorities (it stores the priorities in a vector). The  task T45809 is to
 * implement a stack for the case where only a few nodes have priorities.
 * */
template <typename TNode, typename TPriority>
class BaseStackWithManyPriorities {
public:
  /**
   * \param n The number of nodes in the graph
   *
   * \param ps The priorities for all nodes which do not have the default
   *           prioritiy of zero
   * */
  BaseStackWithManyPriorities(uint64_t n,
                              const Priorities<TNode, TPriority> &ps)
      : sparsePriorities(n, TPriority(0.0)) {
    for (const auto &p : ps) {
      const auto node        = std::get<0>(p);
      const auto priority    = std::get<1>(p);
      sparsePriorities[node] = priority;
    }
  }

  bool empty() const { return ready.empty(); }

protected:
  // The stack used in Kahn's algorithm.
  Priorities<TNode, TPriority> ready;

  // A vector containing all the priorities, with the default (zero) for all
  // nodes without explicit values.
  std::vector<TPriority> sparsePriorities;
};

template <typename TNode, typename TPriority>
class BaseFiloStackWithManyPriorities
    : public BaseStackWithManyPriorities<TNode, TPriority> {

public:
  BaseFiloStackWithManyPriorities(uint64_t n,
                                  const Priorities<TNode, TPriority> &ps)
      : BaseStackWithManyPriorities<TNode, TPriority>(n, ps) {}

  // pop the highest priority node from the stack, or the node nearest the
  // **BACK** if multiple of equal priority.
  TNode basePop() {

    auto &r       = this->ready;
    uint64_t best = 0;
    for (uint64_t i = 1; i < r.size(); ++i) {
      if (std::get<1>(r[i]) >= std::get<1>(r[best])) {
        best = i;
      }
    }
    auto popped = std::get<0>(r[best]);
    r.erase(r.cbegin() + best);
    return popped;
  }
};

template <typename TNode, typename TPriority>
class BaseFifoStackWithManyPriorities
    : public BaseStackWithManyPriorities<TNode, TPriority> {

public:
  BaseFifoStackWithManyPriorities(uint64_t n,
                                  const Priorities<TNode, TPriority> &ps)
      : BaseStackWithManyPriorities<TNode, TPriority>(n, ps) {}

  // pop the highest priority node from the stack, or the node nearest the
  // **FRONT** if multiple of equal priority.
  TNode basePop() {
    auto &r       = this->ready;
    uint64_t best = 0;
    for (uint64_t i = 1; i < r.size(); ++i) {
      if (std::get<1>(r[i]) > std::get<1>(r[best])) {
        best = i;
      }
    }
    auto popped = std::get<0>(r[best]);
    r.erase(r.cbegin() + best);
    return popped;
  }
};

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
