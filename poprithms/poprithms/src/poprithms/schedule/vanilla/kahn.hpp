// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_KAHN_HPP
#define POPRITHMS_SCHEDULE_VANILLA_KAHN_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <schedule/vanilla/error.hpp>

#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

/**
 * The implementations of the various kahn algorithms make heavy use of
 * templating. An arguably more intuitive implementation would use c++
 * polymorphism, but the overhead of virtual methods would cause niticeablely
 * worse performance than the templating approach taken.
 *
 *
 * The functions are templated by:
 *
 * 1) node type,
 * 2) priority type,
 * 3) stack type.
 * */

template <typename T> bool valid(T x, uint64_t end) {
  return x < static_cast<T>(end);
}

template <> inline bool valid(int64_t x, uint64_t end) {
  return x >= 0 && static_cast<uint64_t>(x) < end;
}

template <typename T> void verifyEdges(const Edges<T> &fwdEdges) {
  const auto N = fwdEdges.size();
  for (uint64_t start = 0; start < N; ++start) {
    for (auto end : fwdEdges[start]) {
      if (!valid<T>(end, N)) {
        std::ostringstream oss;
        oss << "Invalid edge (" << start << "->" << end << ") in graph with "
            << N << " nodes. ";
        throw error(oss.str());
      }
    }
  }
}

// return the number of input edges for each node.
template <typename TNode>
std::vector<uint64_t> getOutstandingCount(const Edges<TNode> &fwdEdges) {

  // The total number of nodes in the graph.
  const auto N = fwdEdges.size();

  // Count the number of dependencies each node has
  std::vector<uint64_t> nOutstandingDeps(N, 0);
  for (uint64_t from = 0; from < N; ++from) {
    for (const auto to : fwdEdges[from]) {
      ++nOutstandingDeps[to];
    }
  }
  return nOutstandingDeps;
}

template <typename TNode, typename Stack>
std::vector<TNode> stackBasedKahn(const Edges<TNode> &fwdEdges,
                                  Stack &ready) {

  // assert that the stack is empty.
  if (!ready.empty()) {
    throw error("Expected empty ready-stack at beginning of stackBasedKahn");
  }

  const auto N          = fwdEdges.size();
  auto nOutstandingDeps = getOutstandingCount(fwdEdges);

  // Get the nodes which have no dependencies, they're ready to go into the
  // schedule. Put them on the stack.
  for (uint64_t i = 0; i < N; ++i) {
    if (nOutstandingDeps[i] == 0) {
      ready.push(i);
    }
  }

  std::vector<TNode> schedule;
  schedule.reserve(N);
  while (!ready.empty()) {
    const auto nxt = ready.pop();
    schedule.push_back(nxt);
    for (const auto to : fwdEdges[nxt]) {
      --nOutstandingDeps[to];
      if (nOutstandingDeps[to] == 0) {
        ready.push(to);
      }
    }
  }

  return schedule;
}

// Mapping between a graph with nodes, and links between nodes; and a graph
// where all linked nodes are collapsed/compressed into single nodes.
template <typename TNode> struct LinkMap {

  // map from the larger graph with linked nodes, to the smaller graph where
  // all linked nodes are collapsed into a single node.
  std::vector<TNode> toCompressed;

  std::vector<std::vector<TNode>> toExpanded;

  TNode compressed(TNode i) const { return toCompressed[i]; }
  const std::vector<TNode> &expanded(TNode i) const { return toExpanded[i]; }

  uint64_t nCompressed() const { return toExpanded.size(); }
  uint64_t nExpanded() const { return toCompressed.size(); }
};

template <typename TNode>
LinkMap<TNode> getToLinked(uint64_t nNodes, const Links<TNode> &links) {

  struct Record {
    bool hasFwd{false};
    bool hasBwd{false};
    TNode fwd;
  };

  std::vector<Record> records(nNodes);

  for (const auto &l : links) {
    records[std::get<0>(l)].hasFwd = true;
    records[std::get<0>(l)].fwd    = std::get<1>(l);
    records[std::get<1>(l)].hasBwd = true;
  }

  TNode current{0};
  std::vector<TNode> toCompressed(nNodes, std::numeric_limits<TNode>::max());
  std::vector<std::vector<TNode>> toExpanded;

  for (uint64_t node = 0; node < nNodes; ++node) {
    if (!records[node].hasBwd) {
      toCompressed[node] = current;
      toExpanded.push_back(std::vector<TNode>{TNode(node)});
      auto nodeStar = node;
      while (records[nodeStar].hasFwd) {
        nodeStar = records[nodeStar].fwd;
        toExpanded.back().push_back(nodeStar);
        toCompressed[nodeStar] = current;
      }
      ++current;
    }
  }

  return {toCompressed, toExpanded};
}

template <typename TNode>
Edges<TNode> getCompressedEdges(const LinkMap<TNode> &lm,
                                const Edges<TNode> &fwdEdges) {

  std::vector<std::vector<TNode>> compressedEdges(lm.nCompressed());

  for (uint64_t node0 = 0; node0 < fwdEdges.size(); ++node0) {
    auto compressedNode0 = lm.compressed(node0);
    auto &source         = compressedEdges[compressedNode0];
    for (auto node1 : fwdEdges[node0]) {
      auto dst = lm.compressed(node1);
      if (std::find(source.cbegin(), source.cend(), dst) == source.cend()) {
        if (compressedNode0 != dst) {
          source.push_back(dst);
        }
      }
    }
  }

  return compressedEdges;
}

template <typename TNode>
std::vector<TNode>
expandedSchedule(const LinkMap<TNode> &lm,
                 const std::vector<TNode> &compressedSchedule) {

  std::vector<TNode> expandedSchedule_;
  expandedSchedule_.reserve(lm.nExpanded());
  for (auto node : compressedSchedule) {
    const auto &c = lm.expanded(node);
    expandedSchedule_.insert(expandedSchedule_.end(), c.cbegin(), c.cend());
  }

  return expandedSchedule_;
}

// the priority of a node in the compressed graph is the mean of all the
// non-default priorities in the corresponding nodes in the expanded graph, or
// zero if there are no corresponding nodes with non-default priorities.
template <typename TNode, typename TPriority>
Priorities<TNode, TPriority>
getCompressedPriorities(const LinkMap<TNode> &lm,
                        const Priorities<TNode, TPriority> &priorities) {

  if (priorities.empty()) {
    return priorities;
  }

  std::vector<std::vector<TPriority>> ps(lm.nCompressed());
  for (const auto &p : priorities) {
    ps[lm.compressed(std::get<0>(p))].push_back(std::get<1>(p));
  }
  Priorities<TNode, TPriority> compressedPriorities;
  for (uint64_t i = 0; i < ps.size(); ++i) {
    if (!ps[i].empty()) {
      compressedPriorities.push_back(
          {i,
           std::accumulate(ps[i].cbegin(), ps[i].cend(), TPriority(0.)) /
               TPriority(ps[i].size())});
    }
  }

  return compressedPriorities;
}

// If there are no nodes with priorities, the scheduler
// 'SchedulerWithoutPriorities' is used. This switching between schedulers
// based on the number of priorities allows for faster implementations in the
// case where there are no priorities.
//
// Currently there are only 2 schedulers: 1 for when there are priorities, and
// 1 for when there aren't. In the future we may consider a third scheduler,
// for the case where there a only a few (<<N) nodes with prioritie. See
// T45809.
template <typename TNode,
          typename TPriority,
          class SchedulerWithoutPriorities,
          class SchedulerWithPriorities,
          class... Args>
std::vector<TNode>
linklessDelegate(const Edges<TNode> &fwdEdges,
                 const Priorities<TNode, TPriority> &priorities,
                 const Args &... args) {

  return priorities.empty()
             ? SchedulerWithoutPriorities::kahn(fwdEdges, args...)
             : SchedulerWithPriorities::kahn(fwdEdges, priorities, args...);
}

// If there are links in the graph, (1) a compressed graph without links is
// created, then (2) scheduled, then (3) unpacked/expanded back the to
// original graph with links. If the graph has no links, then it is scheduled
// as is.
//
// This method also handles the case of ErrorIfCycle::Yes and
// VerifyEdges::Yes.
//
// The argument 'getCompressedArgs' is required if the scheduler-specific
// arguments need to be transformed when converting to the compressed graph.
template <typename TNode,
          typename TPriority,
          class SchedulerWithoutPriorities,
          class SchedulerWithPriorities,
          class GetLinkedArgs,
          class... Args>
std::vector<TNode> delegate(const Edges<TNode> &fwdEdges,
                            ErrorIfCycle eic,
                            VerifyEdges ve,
                            const Priorities<TNode, TPriority> &priorities,
                            const Links<TNode> &links,
                            const GetLinkedArgs &getCompressedArgs,
                            const Args &... args) {

  if (ve == VerifyEdges::Yes) {
    verifyEdges<TNode>(fwdEdges);
  }

  std::vector<TNode> schedule;

  auto getLinkless = [&fwdEdges, &priorities, &args...]() {
    return linklessDelegate<TNode,
                            TPriority,
                            SchedulerWithoutPriorities,
                            SchedulerWithPriorities,
                            Args...>(fwdEdges, priorities, args...);
  };

  if (!links.empty()) {
    auto lm                   = getToLinked(fwdEdges.size(), links);
    auto compressedEdges      = getCompressedEdges(lm, fwdEdges);
    auto compressedPriorities = getCompressedPriorities(lm, priorities);
    auto linkedArgsTuple      = getCompressedArgs(lm.toCompressed);

    auto delegateLinked = [&compressedEdges, &compressedPriorities](
                              const Args &... linkedArgs) {
      return linklessDelegate<TNode,
                              TPriority,
                              SchedulerWithoutPriorities,
                              SchedulerWithPriorities,
                              Args...>(
          compressedEdges, compressedPriorities, linkedArgs...);
    };

    // Some C++17 magic to handle parameter packs.
    auto compressedSchedule = std::apply(delegateLinked, linkedArgsTuple);
    schedule                = expandedSchedule(lm, compressedSchedule);
  }

  else {
    schedule = getLinkless();
  }

  const auto N = fwdEdges.size();
  if (eic == ErrorIfCycle::Yes && schedule.size() != N) {
    std::ostringstream oss;
    oss << "Only " << schedule.size() << " of " << N
        << " nodes are scheduled, there is a cycle in the graph. "
        << "The graph has " << priorities.size()
        << " priorities set, and it has " << links.size() << " links. ";
    if (links.size() > 0) {
      auto nScheduleWithoutLinks = getLinkless().size();
      oss << "With all links removed (ignored), " << nScheduleWithoutLinks
          << " of the " << N << " nodes are scheduled. ";
      if (nScheduleWithoutLinks == N) {
        oss << "This suggests that the links are the 'cause' of the cycle. ";
      }
    }
    throw error(oss.str());
  }

  return schedule;
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
