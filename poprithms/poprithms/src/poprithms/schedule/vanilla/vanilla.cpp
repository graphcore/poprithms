// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <schedule/vanilla/error.hpp>
#include <schedule/vanilla/fifostack.hpp>
#include <schedule/vanilla/filostack.hpp>
#include <schedule/vanilla/greedystack.hpp>
#include <schedule/vanilla/kahn.hpp>
#include <schedule/vanilla/randomstack.hpp>

#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

template <typename TNode, typename Stack>
bool isComplete(const Edges<TNode> &es, VerifyEdges ve) {
  if (ve == VerifyEdges::Yes) {
    verifyEdges(es);
  }
  auto outstandingCount = getOutstandingCount(es);
  uint64_t nScheduled{0};
  Stack ready;
  for (uint64_t i = 0; i < es.size(); ++i) {
    if (outstandingCount[i] == 0) {
      ready.push(i);
    }
  }

  // we keep scheduling while the stack is 'valid'. The definition of 'valid'
  // depends on the task at hand.
  while (ready.valid()) {
    auto address = ready.pop();
    ++nScheduled;
    for (auto cAddress : es[address]) {
      --outstandingCount[cAddress];
      if (outstandingCount[cAddress] == 0) {
        ready.push(cAddress);
      }
    }
  }

  return es.size() == nScheduled;
}

template <class TNode> struct BaseQueryStack {
public:
  void push(TNode t) { ready.push_back(t); }
  TNode pop() {
    auto x = ready.back();
    ready.pop_back();
    return x;
  }

  std::vector<TNode> ready;
};

// For checking if a graph is schedulable, we don't stop scheduling til the
// stack is empty.
template <class TNode>
struct IsSchedulableStack : public BaseQueryStack<TNode> {
  bool valid() const { return !this->ready.empty(); }
};

template <typename TNode>
bool Query<TNode>::isSchedulable(const Edges<TNode> &es, VerifyEdges ve) {
  return isComplete<TNode, IsSchedulableStack<TNode>>(es, ve);
}

template <typename TNode>
bool Query<TNode>::isSchedulable(const Edges<TNode> &es,
                                 const Links<TNode> &links,
                                 VerifyEdges ve) {
  if (links.empty()) {
    return isSchedulable(es, ve);
  }
  const auto sch =
      Scheduler<TNode, double>::fifo(es, {}, links, ErrorIfCycle::No, ve);
  return sch.size() == es.size();
}

// For checking if a graph is uniquely schedulable, we stop scheduling as soon
// as either the stack is empty, or it has more than one node in it. More than
// one node implies more than more possible schedule.
template <class TNode>
struct HasUniqueSchedule : public BaseQueryStack<TNode> {
public:
  bool valid() const { return this->ready.size() == 1; }
};

template <typename TNode>
bool Query<TNode>::hasUniqueSchedule(const Edges<TNode> &fwdEdges,
                                     VerifyEdges ve) {
  return isComplete<TNode, HasUniqueSchedule<TNode>>(fwdEdges, ve);
}

template <typename TNode, typename TPriority>
std::vector<TNode> Scheduler<TNode, TPriority>::filo(
    const Edges<TNode> &fwdEdges,
    const Priorities<TNode, TPriority> &priorities,
    const Links<TNode> &links,
    ErrorIfCycle eic,
    VerifyEdges ve) {
  return filo::kahn<TNode, TPriority>(fwdEdges, priorities, links, eic, ve);
}

template <typename TNode, typename TPriority>
std::vector<TNode> Scheduler<TNode, TPriority>::fifo(
    const Edges<TNode> &fwdEdges,
    const Priorities<TNode, TPriority> &priorities,
    const Links<TNode> &links,
    ErrorIfCycle eic,
    VerifyEdges ve) {
  return fifo::kahn<TNode, TPriority>(fwdEdges, priorities, links, eic, ve);
}

template <typename TNode, typename TPriority>
std::vector<TNode> Scheduler<TNode, TPriority>::random(
    const Edges<TNode> &fwdEdges,
    const Priorities<TNode, TPriority> &priorities,
    const Links<TNode> &links,
    uint32_t seed,
    ErrorIfCycle eic,
    VerifyEdges ve) {
  return random::kahn<TNode, TPriority>(
      fwdEdges, priorities, links, seed, eic, ve);
}

std::vector<int64_t>
getSchedule_i64(const std::vector<std::vector<int64_t>> &fwdEdges,
                ErrorIfCycle eic,
                VerifyEdges ve) {
  return Scheduler<int64_t, double>::filo(fwdEdges, {}, {}, eic, ve);
}

std::vector<uint64_t>
getSchedule_u64(const std::vector<std::vector<uint64_t>> &fwdEdges,
                ErrorIfCycle eic,
                VerifyEdges ve) {
  return Scheduler<uint64_t, double>::filo(fwdEdges, {}, {}, eic, ve);
}

template <typename TNode, typename TPriority, typename TAllocSize>
std::vector<TNode> GreedyScheduler<TNode, TPriority, TAllocSize>::kahn(
    const Edges<TNode> &fwdEdges,
    const Priorities<TNode, TPriority> &priorities,
    const Links<TNode> &links,
    const std::vector<TAllocSize> &sizes,
    const Edges<TNode> &allocsToNodes,
    ErrorIfCycle eic,
    VerifyEdges ve) {
  return greedy::kahn<TNode, TPriority>(
      fwdEdges, priorities, links, sizes, allocsToNodes, eic, ve);
}

template class Query<int64_t>;
template class Query<uint64_t>;

template class Scheduler<int64_t, double>;
template class Scheduler<uint64_t, double>;

template class GreedyScheduler<int64_t, double, int>;
template class GreedyScheduler<uint64_t, double, int>;

} // namespace vanilla
} // namespace schedule
} // namespace poprithms
