// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_GREEDYSTACK
#define POPRITHMS_SCHEDULE_VANILLA_GREEDYSTACK

#include <vector>

#include <schedule/vanilla/error.hpp>
#include <schedule/vanilla/kahn.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {
namespace greedy {

/**
 * 'greedy' is the namespace for a kahn tie-breaker which chooses the Op which
 * results in the largest immediate liveness reduction at every step.
 *
 * The implementation of the algorithm is 'stack based', which means it uses
 * the stackBasedKahn template function.
 * */

template <typename TNode, typename TAllocSize> class BaseStack {
public:
  using TNodes      = std::vector<TNode>;
  using TAllocSizes = std::vector<TAllocSize>;

protected:
  // The standarad kahn algorithm's 'ready' stack.
  TNodes ready;

  // The sizes of the allocations:
  TAllocSizes allocSizes;

  // map from allocations to the nodes which require them:
  std::vector<TNodes> allocsToNodes;

  // map from nodes to allocations:
  std::vector<std::vector<uint64_t>> nodesToAllocs;

  // the number of unscheduled nodes for each allocation:
  std::vector<int> nOutstandingForAlloc;

  // true if there is at least 1 scheduled node for an alloc:
  std::vector<bool> allocIsLive;

public:
  uint64_t nAllocs() const { return allocSizes.size(); }

  BaseStack(uint64_t n,
            const TAllocSizes &allocSizes_,
            const std::vector<TNodes> &allocsToNodes_)
      : allocSizes(allocSizes_), allocsToNodes(allocsToNodes_) {
    if (allocSizes_.size() != allocsToNodes_.size()) {
      std::ostringstream oss;
      oss << "Ambiguous number of allocations: " << allocSizes_.size()
          << "sizes provided, but only " << allocsToNodes_.size()
          << "mappings from allocations to ops. ";
      throw error(oss.str());
    }
    nodesToAllocs.resize(n);
    nOutstandingForAlloc.reserve(n);
    for (uint64_t a = 0; a < nAllocs(); ++a) {
      for (const auto &b : allocsToNodes[a]) {
        nodesToAllocs[b].push_back(a);
      }
      nOutstandingForAlloc.push_back(allocsToNodes[a].size());
    }
    allocIsLive.resize(nAllocs(), false);
  }

  // If node 't' were to be scheduled, how much would the liveness change by?
  TAllocSize deltaLive(TNode t) const {
    TAllocSize delta = TAllocSize(0.);
    for (auto a : nodesToAllocs[t]) {
      if (nOutstandingForAlloc[a] == 1) {
        delta -= allocSizes[a];
      }
      if (!allocIsLive[a]) {
        delta += allocSizes[a];
      }
    }
    return delta;
  }

  bool empty() const { return ready.empty(); }

  void push(TNode t) { ready.push_back(t); }

  TNode basePop(uint64_t readyIndex) {
    auto node = ready[readyIndex];
    for (auto a : nodesToAllocs[node]) {
      allocIsLive[a] = true;
    }
    ready.erase(ready.cbegin() + readyIndex);
    for (const auto &a : nodesToAllocs[node]) {
      --nOutstandingForAlloc[a];
    }
    return node;
  }
};

template <typename TNode, typename TAllocSize>
class StackWithoutPriorities : public BaseStack<TNode, TAllocSize> {
public:
  using TNodes      = std::vector<TNode>;
  using TAllocSizes = std::vector<TAllocSize>;

  StackWithoutPriorities(uint64_t n,
                         const TAllocSizes &allocSizes_,
                         const std::vector<TNodes> &allocsToNodes_)
      : BaseStack<TNode, TAllocSize>(n, allocSizes_, allocsToNodes_) {}

  uint64_t getBest() const {
    const auto &r = this->ready;

    std::vector<TAllocSize> deltas;
    deltas.reserve(r.size());
    // we compute the change in liveness for each node in the ready stack.
    // Note that we musst recompute this value for every node, as it is not
    // constant but rather changes depending on what has already been
    // scheduled.
    for (auto node : r) {
      deltas.push_back(this->deltaLive(node));
    }
    uint64_t best = 0;
    for (uint64_t node = 1; node < r.size(); ++node) {
      if (deltas[node] <= deltas[best]) {
        best = node;
      }
    }
    return best;
  }

  TNode pop() { return this->basePop(getBest()); }

private:
};

template <typename TNode, typename TAllocSize>
class SchedulerWithoutPriorities {
public:
  using TNodes      = std::vector<TNode>;
  using TAllocSizes = std::vector<TAllocSize>;
  static std::vector<TNode> kahn(const Edges<TNode> &fwdEdges,
                                 const TAllocSizes &allocSizes_,
                                 const std::vector<TNodes> &allocsToNodes_) {
    auto a_ =
        StackWithoutPriorities(fwdEdges.size(), allocSizes_, allocsToNodes_);
    return stackBasedKahn<TNode>(fwdEdges, a_);
  }
};

template <typename TNode, typename TPriority, typename TAllocSize>
class StackWithManyPriorities : public BaseStack<TNode, TAllocSize> {
public:
  using typename BaseStack<TNode, TAllocSize>::TAllocSizes;
  using TP = std::tuple<TNode, TPriority>;
  StackWithManyPriorities(uint64_t n,
                          const std::vector<TP> &priorities,
                          const TAllocSizes &allocSizes_,
                          const Edges<TNode> &allocsToNodes_)
      : BaseStack<TNode, TAllocSize>(n, allocSizes_, allocsToNodes_),
        sparsePriorities(n, TPriority(0.0)) {
    for (const auto &p : priorities) {
      sparsePriorities[std::get<0>(p)] = std::get<1>(p);
    }
  }

  uint64_t getBest() const {

    struct Triplet {
      uint64_t index     = uint64_t(0);
      TPriority priority = TPriority(0);
      TAllocSize delta   = TAllocSize(0);

      // First check priorities, and if they are the same then check changes
      // in liveness.
      bool betterThan(const Triplet &rhs) const {
        if (priority != rhs.priority) {
          return priority > rhs.priority;
        }
        return delta < rhs.delta;
      }
    };

    Triplet current;

    auto updateCurrent = [&current, this](uint64_t index) {
      current.index    = index;
      current.priority = sparsePriorities[this->ready[index]];
      current.delta    = this->deltaLive(this->ready[index]);
    };

    updateCurrent(0);
    auto best = current;

    for (uint64_t node = 1; node < this->ready.size(); ++node) {
      updateCurrent(node);
      if (current.betterThan(best)) {
        best = current;
      }
    }
    return best.index;
  }

  TNode pop() { return this->basePop(getBest()); }

private:
  std::vector<TPriority> sparsePriorities;
};

template <typename TNode, typename TPriority, typename TAllocSize>
class SchedulerWithManyPriorities {
public:
  std::vector<TNode> static kahn(
      const Edges<TNode> &fwdEdges,
      const Priorities<TNode, TPriority> &priorities,
      const std::vector<TAllocSize> &sizes,
      const Edges<TNode> &allocsToNodes) {
    auto a_ = StackWithManyPriorities<TNode, TPriority, TAllocSize>(
        fwdEdges.size(), priorities, sizes, allocsToNodes);
    return stackBasedKahn<TNode>(fwdEdges, a_);
  }
};

template <typename TNode, typename TPriority, typename TAllocSize>
std::vector<TNode> kahn(const Edges<TNode> &fwdEdges,
                        const Priorities<TNode, TPriority> &priorities,
                        const Links<TNode> &links,
                        const std::vector<TAllocSize> &sizes,
                        const Edges<TNode> &allocsToNodes,
                        ErrorIfCycle eic,
                        VerifyEdges ve) {

  // If the graph has links in it, it is reduced/compressed to a graph without
  // any links. The allocations need to be remapped to the nodes in the
  // compressed graph. This method does this remapping:
  auto getLinkedArgs = [&sizes,
                        &allocsToNodes](const std::vector<TNode> &toLinked) {
    const auto &linkedSizes = sizes;
    Edges<TNode> linkedAllocsToNodes;
    linkedAllocsToNodes.reserve(allocsToNodes.size());
    for (const auto &allocToNodes : allocsToNodes) {
      linkedAllocsToNodes.push_back({});
      auto &a = linkedAllocsToNodes.back();
      for (const auto &unlinkedNode : allocToNodes) {
        const auto linkedNode = toLinked[unlinkedNode];
        if (std::find(a.cbegin(), a.cend(), linkedNode) == a.cend()) {
          a.push_back(linkedNode);
        }
      }
    }

    return std::tuple<const std::vector<TAllocSize> &, Edges<TNode>>{
        linkedSizes, linkedAllocsToNodes};
  };

  return delegate<TNode,
                  TPriority,
                  SchedulerWithoutPriorities<TNode, TAllocSize>,
                  SchedulerWithManyPriorities<TNode, TPriority, TAllocSize>>(
      fwdEdges,
      eic,
      ve,
      priorities,
      links,
      getLinkedArgs,
      sizes,
      allocsToNodes);
}

} // namespace greedy
} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
