// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <numeric>

#include <poprithms/schedule/connectedcomponents/connectedcomponents.hpp>
#include <poprithms/schedule/shift/allocsimplifier.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

bool AllocSimplifier::combineAllocsWithCommonOps(Graph &graph) {

  // toMerge[x] will contain all allocations with the same op pattern as x,
  // where x is the lowest in the partition.
  std::vector<std::vector<AllocAddress>> toMerge(graph.nAllocs());

  auto group = [&graph](AllocAddress a) {
    const auto &ops = graph.getAlloc(a).getOps();
    if (ops.size() == 0) {
      return a;
    }
    const auto &op = ops[0];
    for (const auto &alloc1 : graph.getOp(op).getAllocs()) {
      if (alloc1 < a) {
        if (ops == graph.getAlloc(alloc1).getOps()) {
          return alloc1;
        }
      }
    }
    return a;
  };

  for (uint64_t a = 0; a < graph.nAllocs(); ++a) {
    toMerge[group(a)].push_back(a);
  }

  bool changed{false};
  for (const auto &g : toMerge) {
    if (g.size() > 1 && graph.getAlloc(g[0]).nOps() > 0) {
      changed = true;

      // The combined weight of all the allocs which share #opsWithCommon.
      AllocWeight combined = graph.getAlloc(g[0]).getWeight();
      for (uint64_t i = 1; i < g.size(); ++i) {
        const auto &toRemove = graph.getAlloc(g[i]);
        combined += toRemove.getWeight();
        graph.disconnectAlloc(toRemove.getAddress());
      }

      graph.updateWeight(g[0], combined);
    }
  }
  return changed;
}

bool AllocSimplifier::disconnectAllocsWithOneOp(Graph &graph) {

  bool changed{false};

  for (auto &alloc : graph.getAllocs()) {
    if (alloc.nOps() == 1) {
      graph.disconnectAlloc(alloc.getAddress());
      changed = true;
    }
  }
  return changed;
}

bool AllocSimplifier::disconnectAllocsWithZeroWeight(Graph &graph) {

  bool changed{false};

  for (const auto &alloc : graph.getAllocs()) {
    if (alloc.nOps() > 0 && alloc.getWeight() == AllocWeight::zero()) {
      changed = true;
      graph.disconnectAlloc(alloc.getAddress());
    }
  }
  return changed;
}

bool AllocSimplifier::disconnectInbetweenerAllocs(
    Graph &graph,
    const TransitiveClosure &closure) {

  bool changed{false};

  for (const auto &alloc : graph.getAllocs()) {
    const auto &ops     = alloc.getOps();
    const auto statuses = closure.getExtremumStatuses(ops);
    std::vector<OpAddress> toRemove;
    std::vector<OpAddress> toKeep;
    for (uint64_t i = 0; i < ops.size(); ++i) {
      const auto s = statuses[i];
      if (std::get<0>(s) == transitiveclosure::IsFirst::No &&
          std::get<1>(s) == transitiveclosure::IsFinal::No) {
        toRemove.push_back(ops[i]);
      } else {
        toKeep.push_back(ops[i]);
      }
    }
    if (!toRemove.empty()) {
      changed = true;
      graph.update(alloc.getAddress(), toKeep, toRemove);
    }
  }
  return changed;
}

bool AllocSimplifier::disconnectFixedDurationAllocs(
    Graph &graph,
    const TransitiveClosure &closure) {

  uint64_t nChanged{0};

  for (const auto &alloc : graph.getAllocs()) {
    if (alloc.nOps() > 0) {
      const auto &ops   = alloc.getOps();
      const auto bounds = closure.getDurationBound(ops);
      if (bounds.high == bounds.low + 1) {
        graph.disconnectAlloc(alloc.getAddress());
        ++nChanged;
      }
    }
  }

  return nChanged != 0;
}

bool AllocSimplifier::connectContiguousAllocs(
    Graph &graph,
    const TransitiveClosure &closure) {

  bool changed{false};

  for (const auto &op : graph.getOps()) {
    std::vector<AllocAddress> firstWithRespectTo;
    std::vector<AllocAddress> finalWithRespectTo;
    for (const auto allocAddress : op.getAllocs()) {
      const auto &allocOps = graph.getAlloc(allocAddress).getOps();
      const auto stat = closure.getExtremumStatus(op.getAddress(), allocOps);
      if (std::get<0>(stat) == transitiveclosure::IsFirst::Yes) {
        firstWithRespectTo.push_back(allocAddress);
      }
      if (std::get<1>(stat) == transitiveclosure::IsFinal::Yes) {
        finalWithRespectTo.push_back(allocAddress);
      }
    }

    std::vector<std::array<AllocAddress, 2>> toMerge;
    const auto present = [&toMerge](OpAddress a) {
      for (const auto &x : toMerge) {
        if (std::get<0>(x) == a || std::get<1>(x) == a) {
          return true;
        }
      }
      return false;
    };

    const auto sameWeight = [&graph](AllocAddress a, AllocAddress b) {
      return (graph.getAlloc(a).getWeight() - graph.getAlloc(b).getWeight() ==
              AllocWeight::zero());
    };

    for (auto firstWrt : firstWithRespectTo) {
      for (auto finalWrt : finalWithRespectTo) {
        if (sameWeight(firstWrt, finalWrt) && !present(firstWrt) &&
            !present(finalWrt)) {
          toMerge.push_back({firstWrt, finalWrt});
        }
      }
    }

    if (!toMerge.empty()) {
      changed = true;

      // replace alloc1 everywhere with alloc0.
      for (const auto &[alloc0, alloc1] : toMerge) {
        if (alloc0 != alloc1) {
          for (auto op_ : graph.getAlloc(alloc1).getOps()) {
            graph.insertOpAlloc(op_, alloc0);
          }
        }
        graph.disconnectAlloc(alloc1);
      }
    }
  }

  return changed;
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
