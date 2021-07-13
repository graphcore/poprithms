// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_TRAVERSAL_HPP
#define POPRITHMS_COMMON_MULTIOUT_TRAVERSAL_HPP

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * Traverse through the Ops in graph #g in a forwards direction, starting at
 * the Tensors in #starts. During the traversal, record all OpTraversals
 * taken. Terminate, and do not record, OpTraversals for which #accept
 * evaluates as false.
 *
 * This is similar to depth-first search of all Ops in a graph, except that it
 * records all the ways in which the Ops can be entered and exited.
 *
 * \param G g The graph to traverse.
 *
 * \param starts The starting Tensors for the depth-first traversal.
 *
 * \param AcceptanceCondition accept. The condition which an OpTraversal must
 *                                    evaluate true on to be included in the
 *                                    final returned set, and to continue the
 *                                    forwards traversal.
 *
 * \return the OpTraversals travelled, returned unique and sorted by
 *         OpTraversal::operator<
 *
 * */
template <class G, class AcceptanceCondition>
std::vector<OpTraversal> depthFirstForward(const G &g,
                                           const TensorIds &starts,
                                           AcceptanceCondition &&accept) {

  // The stack of Tensors from which we still need to traverse forwards from,
  // through all consumers:
  TensorIds toProcess = starts;

  // We keep track of all Tensors visited (processed) so that we do not repeat
  // traversals.
  std::set<TensorId> visited{starts.cbegin(), starts.cend()};

  // The set of all traversals taken.
  std::vector<OpTraversal> traversals;

  while (!toProcess.empty()) {
    auto nxt = toProcess.back();
    toProcess.pop_back();

    // For all consumers, and all outputs of consumers, check if the
    // OpTraversal is acceptable:
    for (auto cId : g.consumptionIds(nxt)) {
      for (uint64_t out = 0; out < g.nOutTensors(cId.opId()); ++out) {
        OpTraversal route{cId.inIndex(), cId.opId(), out};
        if (accept(route)) {

          // we may be inserting a duplicate here, but we'll sort this out
          // just before returning.
          traversals.push_back(route);
          TensorId tId{cId.opId(), out};
          if (visited.find(tId) == visited.cend()) {
            visited.emplace(tId);
            toProcess.push_back(tId);
          }
        }
      }
    }
  }
  return util::unisorted(traversals);
}

template <class AcceptanceCondition> class SearchUntilOneFound {
public:
  SearchUntilOneFound(const TensorId &target_, AcceptanceCondition &&accept_)
      : target(target_), accept(std::move(accept_)), found_(false) {}

  bool found() const { return found_; }

  bool isTarget(const OpTraversal &ot) const {
    return ot.opId() == target.opId() && ot.outIndex() == target.outIndex();
  }

  bool operator()(const OpTraversal &ot) {
    // untraversable
    if (!accept(ot)) {
      return false;
    }

    // If the target tensor has already been found, return false to terminate
    // searches along any new paths.
    if (found()) {
      return false;
    }
    if (isTarget(ot)) {
      found_ = true;
    }
    return true;
  }

private:
  TensorId target;
  AcceptanceCondition accept;
  bool found_;
};

/**
 * Starting from the tensors in #starts, is it possible to traverse the graph
 * #g to the tensor #target along a path of traversals which are all accepted
 * by the AcceptanceCondition #accept?
 * */
template <class G, class AcceptanceCondition>
bool isFwdReachable(const G &g,
                    const TensorIds &starts,
                    const TensorId &target,
                    AcceptanceCondition &&accept) {
  SearchUntilOneFound<AcceptanceCondition> hyper(
      target, std::forward<AcceptanceCondition>(accept));

  // Perform a depth-first search, starting at #starts and traversing towards
  // targets. Terminate the traversal as soon as the target is found.
  depthFirstForward(g, starts, hyper);
  return hyper.found();
}

/**
 * Traverse through Ops in #g in a bacwkards direction, starting at the
 * Tensors in #starts, recording all OpTraversals taken during the traversal.
 * Terminate, and do not record, OpTraversals for which #accept evaluates as
 * false.
 *
 * \sa depthFirstForward.
 * */
template <class G, class AcceptanceCondition>
std::vector<OpTraversal> depthFirstBackward(const G &g,
                                            const TensorIds &starts,
                                            AcceptanceCondition &&accept) {

  // unlike depthFirstForward, there is only 1 for loop nested insode the
  // while loop. This asymmetry arises from the fact that Tensors only have 1
  // producer, but multiple consumers. This difference also prevents us from
  // re-using the forwards traversal implementation for the backward case.

  TensorIds toProcess = starts;
  std::set<TensorId> visited{starts.cbegin(), starts.cend()};
  std::vector<OpTraversal> routes;

  while (!toProcess.empty()) {
    auto nxt = toProcess.back();
    toProcess.pop_back();
    const auto inTensors = g.inTensorIds(nxt.opId());
    for (uint64_t i = 0; i < inTensors.size(); ++i) {
      OpTraversal route{i, nxt.opId(), nxt.outIndex()};
      if (accept(route)) {
        routes.push_back(route);
        if (visited.find(inTensors[i]) == visited.cend()) {
          visited.emplace(inTensors[i]);
          toProcess.push_back(inTensors[i]);
        }
      }
    }
  }
  return util::unisorted(routes);
}

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
