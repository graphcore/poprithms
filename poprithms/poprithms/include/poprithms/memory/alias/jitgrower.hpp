// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_JITGROWER_HPP
#define POPRITHMS_MEMORY_ALIAS_JITGROWER_HPP

#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/vanilla/vanillamap.hpp>

namespace poprithms {
namespace memory {
namespace alias {

/**
 * A helper class for creating/extending a memory::alias::Graph for a subset
 * of tensors in another (external) graph.
 *
 * It is sometimes not desirable to map an entire external (popart/popir/etc)
 * graph to a memory::alias::Graph, because only a small subset of the tensors
 * in the external graph might require alias information. This class helps
 * to create a memory::alias::Graph corresponding only to the subset of
 * tensors required, and for it to then be extended when new tensors in the
 * external graph require aliasining information.
 *
 * To use this class, a user must create a new class which inherits from it
 * and override 3 virtual methods,
 *
 * 1) aliasingIns: Describes the DAG structure of the tensors.
 *
 * 2) containsAliasTensor: Describes the current state of the alias graph
 *    - does the underlying alias graph (which this class does not manage)
 *    contain an alias tensor for specific external tensor?
 *
 * 3) growAliasTensors: Describes what nodes should be inserted into the
 *    memory::alias::Graph to correspond to the external graph.
 * */
template <typename ExternalTensorId> class JitGrower {

private:
  using ExternalTensorIds = std::vector<ExternalTensorId>;

  /**
   * return the inputs of #tId which are aliases of #tId. In other words,
   * which tensors is #tId a view-change of? For example
   * if t = concat(init("a"), reverse(init("b"))) then the aliasing inputs of
   * t are init("a") and reverse(init("b")). init("b") has no aliasing inputs.
   * */
  virtual ExternalTensorIds
  aliasingIns(const ExternalTensorId &tId) const = 0;

  /**
   * return true if there is already an alias graph tensor grown for the
   * external tensor #tId.
   * */
  virtual bool containsAliasTensor(const ExternalTensorId &) const = 0;

  /**
   * Grow alias model tensors for each of the external tensors in #scheduled.
   * The tensors are ordered topologically, so that any inputs (\sa
   * aliasingIns) of scheduled[i] which are also in scheduled will appear at
   * scheduled[i'] for some i' < i.
   * */
  virtual void growAliasTensors(const ExternalTensorIds &scheduled) = 0;

public:
  /**
   * Ensure that alias information is available for all tensors in #tIds.
   * */
  void extend(const ExternalTensorIds &tIds) {

    // Perform a depth-first search backwards through the graph, stopping at
    // tensors for which alias information is already available. Once the dfs
    // is complete, grow alias graph equivalents for all the (external
    // project) tensors found in the dfs.

    std::unordered_set<ExternalTensorId> visited;
    ExternalTensorIds traverseStack;

    auto insertTensor = [&visited,
                         &traverseStack](const ExternalTensorId &tId) {
      visited.insert(tId);
      traverseStack.push_back(tId);
    };
    for (const auto &tId : tIds) {
      if (!containsAliasTensor(tId)) {
        insertTensor(tId);
      }
    }
    while (!traverseStack.empty()) {
      const auto nxt = traverseStack.back();
      traverseStack.pop_back();
      for (auto &&tId : aliasingIns(nxt)) {
        if (visited.count(tId) == 0 && !containsAliasTensor(tId)) {
          insertTensor(tId);
        }
      }
    }

    // At this point, the set visited contains all tensors which need to have
    // an alias tensor grown for them. But in what order?
    std::unordered_map<ExternalTensorId, ExternalTensorIds> fwdEdges;
    for (const auto &tId : visited) {
      fwdEdges.insert({tId, {}});
    }

    for (const auto &after : visited) {
      for (auto &&before : aliasingIns(after)) {
        if (visited.count(before) != 0) {
          auto found = fwdEdges.find(before);
          found->second.push_back(after);
        }
      }
    }

    using namespace poprithms::schedule;
    auto sched = vanilla::getSchedule(
        fwdEdges, vanilla::ErrorIfCycle::Yes, vanilla::VerifyEdges::Yes);

    growAliasTensors(sched);

    // Check that this method has done what it promised.
    for (const auto &tId : tIds) {
      if (!containsAliasTensor(tId)) {
        throw poprithms::error::error(
            "memory::alias",
            "Expected all tensors passed to 'extend' to have corresponding "
            "memory::alias::Tensors at this point. Possibly an invalid "
            "virtual method override?");
      }
    }
  }
};

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
