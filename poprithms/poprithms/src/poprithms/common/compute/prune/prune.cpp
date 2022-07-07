// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <set>
#include <sstream>

#include <common/compute/error.hpp>
#include <common/compute/prune/callstackquerier.hpp>
#include <common/compute/prune/prunemutator.hpp>

#include <poprithms/common/compute/prune/pruner.hpp>

namespace poprithms {
namespace common {
namespace compute {

InIndices CallstackQuerier::nonCalleeCopyInIndices(OpId opId) const {
  if (!graph().computeOp(opId).hasCallees()) {
    return graph().computeOp(opId).inIndices();
  }
  return wc(opId).nonCopyToCalleeIndices();
}

std::vector<std::pair<InIndex, TensorId>>
CallstackQuerier::copyInDsts(OpId opId) const {
  auto &&tIds = wc(opId).inTensorIdDsts();
  std::vector<std::pair<InIndex, TensorId>> v;
  v.reserve(tIds.size());
  for (uint64_t i = 0; i < tIds.size(); ++i) {
    v.push_back({i, tIds[i]});
  }
  return v;
}

bool CallstackQuerier::isCarriedTo(const TensorId &tId,
                                   const CallStack &cs) const {
  // empty call stack means this is not inside a repeat, and so is not carried
  // to:
  if (cs.empty()) {
    return false;
  }
  return wc(cs.back().caller()).isCarriedTo(tId);
}

TensorId CallstackQuerier::carriedFrom(const TensorId &tId,
                                       const CallStack &cs) const {
  auto caller = cs.back().caller();
  return wc(caller).carriedFrom(tId);
}

TensorIds Pruner::getUnprunenableRefs(const Graph &graph) {

  // All tensors which have references in other graphs, and all of their
  // aliases:
  auto aliasesWithRefs =
      MemoryAliasMapper(graph, {}).aliases(graph.tensorsWithRefs());

  // All the tensors which might be reachable from the set of runnable
  // sub-graphs:
  auto rfr_ = graph.reachableFromRunnable();
  std::set<SubGraphId> rfr{rfr_.cbegin(), rfr_.cend()};

  TensorIds ids;

  for (auto t0 : aliasesWithRefs) {
    if (rfr.count(graph.subGraphId(t0)) != 0) {

      // The tensor t0 is in a reachable sub-graph, and it is the output of a
      // RefFrom_ op. Ensure that it's root reference is not pruned:
      if (!graph.isRootRef(t0)) {
        ids.push_back(graph.rootRef(t0));
      }

      // The tensor t0 is in a reachable sub-graph, and is aliased to a tensor
      // with a reference in another graph. Check if any of its conusmers
      // modify it. If they do, they the modifier should not be pruned as it
      // could have side effects in a different sub-graph. To prevent the
      // modifier from being pruned, add its outputs to the set of unpruneable
      // tensors.
      for (auto conId : graph.consumptionIds(t0)) {
        auto conOpId = conId.opId();
        if (graph.modifies(conOpId, conId.inIndex())) {
          const auto &modifier = graph.computeOp(conOpId);
          if (modifier.nOutTensors() == 0) {
            std::ostringstream oss;
            oss << "Ops which modify an input should always have an output. ";
            oss << "The op " << modifier << " is therefore invalid.";
            throw error(oss.str());
          }
          for (auto modified : modifier.outTensorIds()) {
            ids.push_back(modified);
          }
        }
      }
    }
  }

  return ids;
}

void Pruner::pruneButPreserveUnpruneableRefs(Graph &graph,
                                             TensorIds toPreserve) {

  // A conservative set of tensors which are determined to be unpruneable. If
  // there are no references between graphs, this is the empty-set.
  auto unpruneableRefs = getUnprunenableRefs(graph);

  // Add the set of globally unpruneable tensors above to the user provided
  // set of tensors to preserve.
  toPreserve.insert(
      toPreserve.cend(), unpruneableRefs.cbegin(), unpruneableRefs.cend());

  // Perform the pruning.
  PruneMutator mm(graph);
  CallstackQuerier mq(graph);
  poprithms::program::prune::Pruner::prune(
      mq, mm, graph.runnable(), toPreserve);
}

} // namespace compute
} // namespace common
} // namespace poprithms
