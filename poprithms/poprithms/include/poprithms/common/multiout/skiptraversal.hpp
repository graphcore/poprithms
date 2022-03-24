// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_SKIP_TRAVERSAL_HPP
#define POPRITHMS_COMMON_MULTIOUT_SKIP_TRAVERSAL_HPP

#include <iostream>

#include <poprithms/common/multiout/traversal.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * Perform a depth-first forward search through a graph #g, which contains
 * 'skips' or 'loop backs': certain tensors have additional forward edges to
 * tensors preceding them in the DAG, which effectively makes the graph
 * cyclical. This method is useful for analyzing graphs in loop/repeat
 * operators which have carry dependencies.
 *
 * \tparam TGraph extends the graph class for #depthFirstForward with the
 *         methods #isCarriedFrom and #carriedTo, which describe the skip
 *        connections. It also has a depth-first method #dfs.
 *
 * \param starts The tensors from which to start for forward traversal.
 *
 * \param rptCount :
 *        The algorithm implemeted is effectively,
 *        for i in range(rptCount):
 *            1) perform standard depth first search from starts
 *            2) set starts to all tensors carried to during traversal
 *        That is, #rptCount controls the number of times the skip edges are
 *        traversed.
 *
 * */
template <class TGraph, class TraversalAcceptanceCondition>
std::set<TensorId>
depthFirstWithSkips(const TGraph &g,
                    const TensorIds &starts,
                    const TraversalAcceptanceCondition &accept,
                    uint64_t rptCount) {

  // Starting from the inputs at #starts, iterate #rptCounts times,
  // recording all tensors traversed by traversals which are valid according
  // to #accept. Stop early (stop iterating) if no new tensors added.

  std::set<TensorId> visited;
  TensorIds nextStarts;
  TensorIds currentStarts;
  for (auto tId : starts) {
    nextStarts.push_back(tId);
  }

  // depth first search, starting from #currentStarts, terminating if #accept
  // is false or if a tensor has already been visited. No carries here,
  // they're handled in the following while loop.
  const auto dfs = [&accept, &g, &visited, &currentStarts]() {
    auto superAccept = [&accept, &visited, &g](const OpTraversal &ot) {
      return accept(ot) && (visited.count(g.outTensorId(ot)) == 0);
    };
    TensorIds locallyVisited;
    for (auto s : currentStarts) {
      locallyVisited.push_back(s);
    }
    // standard depth first search, with the additional termination condition
    // that if a tensor has been visited in a previous iteration then stop.
    for (auto trav : g.dfs(currentStarts, superAccept)) {
      locallyVisited.push_back(g.outTensorId(trav));
    }
    return locallyVisited;
  };

  uint64_t iter{0};
  while (iter < rptCount && !nextStarts.empty()) {
    std::swap(currentStarts, nextStarts);
    nextStarts.clear();
    for (auto t : dfs()) {
      if (visited.count(t) == 0) {
        if (g.isCarriedFrom(t)) {
          nextStarts.push_back(g.carriedTo(t));
        }
        visited.insert(t);
      }
    }
    ++iter;
  }
  return visited;
}

template <class TSkip, class TGraph> class SkipFwdHelper {
public:
  SkipFwdHelper(const TSkip &r, const TGraph &m)
      : skipEdgeHelper(r), graphEdgeHelper(m) {}
  TensorId outTensorId(const OpTraversal &ot) const {
    return graphEdgeHelper.outTensorId(ot);
  }
  bool isCarriedFrom(const TensorId &tId) const {
    return skipEdgeHelper.isCarriedFrom(tId);
  }
  TensorId carriedTo(const TensorId &tId) const {
    return skipEdgeHelper.carriedTo(tId);
  }

  template <class Accept>
  OpTraversals dfs(const TensorIds &starts, Accept &&a) const {
    return poprithms::common::multiout::depthFirstForward<TGraph, Accept>(
        graphEdgeHelper, starts, a);
  }

private:
  const TSkip &skipEdgeHelper;
  const TGraph &graphEdgeHelper;
};

/**
 *
 * */
template <class TSkip, class TGraph, class TraversalAcceptanceCondition>
std::set<TensorId>
depthFirstFwdWithSkips(const TSkip &skipEdgeHelper,
                       const TGraph &graphEdgeHelper,
                       const TensorIds &starts,
                       const TraversalAcceptanceCondition &accept,
                       uint64_t rptCount) {

  SkipFwdHelper<TSkip, TGraph> h(skipEdgeHelper, graphEdgeHelper);
  return depthFirstWithSkips(h, starts, accept, rptCount);
}

template <class TSkip, class TGraph> class SkipBwdHelper {
public:
  SkipBwdHelper(const TSkip &r, const TGraph &m)
      : skipEdgeHelper(r), graphEdgeHelper(m) {}
  TensorId outTensorId(const OpTraversal &ot) const {
    return graphEdgeHelper.inTensorId(ot);
  }
  bool isCarriedFrom(const TensorId &tId) const {
    return skipEdgeHelper.isCarriedTo(tId);
  }
  TensorId carriedTo(const TensorId &tId) const {
    return skipEdgeHelper.carriedFrom(tId);
  }

  template <class Accept>
  OpTraversals dfs(const TensorIds &starts, Accept &&a) const {
    return poprithms::common::multiout::depthFirstBackward<TGraph, Accept>(
        graphEdgeHelper, starts, a);
  }

private:
  const TSkip &skipEdgeHelper;
  const TGraph &graphEdgeHelper;
};

template <class TSkip, class TGraph, class TraversalAcceptanceCondition>
std::set<TensorId>
depthFirstBwdWithSkips(const TSkip &skipEdgeHelper,
                       const TGraph &graphEdgeHelper,
                       const TensorIds &starts,
                       const TraversalAcceptanceCondition &accept,
                       uint64_t rptCount) {

  SkipBwdHelper<TSkip, TGraph> h(skipEdgeHelper, graphEdgeHelper);
  return depthFirstWithSkips(h, starts, accept, rptCount);
}

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
