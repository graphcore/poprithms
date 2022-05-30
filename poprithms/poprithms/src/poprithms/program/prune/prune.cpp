// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <iterator>
#include <limits>
#include <set>
#include <sstream>

#include <poprithms/program/callstack/querier.hpp>
#include <poprithms/program/callstack/stackutil.hpp>
#include <poprithms/program/distributed/program.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace poprithms {
namespace program {
namespace prune {

TensorIds Pruner::unpruneable(const callstack::Querier &querier,
                              const SubGraphIds &callables,
                              const TensorIds &unpruneableBackSources) {
  if (callables.empty()) {
    throw error("Callables is empty. This means that every op and tensor is "
                "unreachable and can therefore be pruned. Is this a user "
                "error? Bailing in case.");
  }

  /**
   * A map with
   *   keys  :  tensor ids
   *   values:  call stacks that the tensor's graph might appear in.
   * */
  auto stackMap = querier.nestedFullStackMap(callables);

  StackTensorIds unpruneableStackIds;
  for (auto unpruneableTensorId : unpruneableBackSources) {
    auto found = stackMap.find(unpruneableTensorId);
    if (found == stackMap.cend()) {
      std::ostringstream oss;
      oss << "Tensor " << unpruneableTensorId
          << " is provided as an unprueable source tensor. "
          << "However, given the set of callable sub-graphs " << callables
          << " it is impossible to reach " << unpruneableTensorId
          << ". Is this a user error? "
          << "It seems strange that a tensor which cannot "
          << "be reached should be tagged as unpruneable."
          << " Bailing in case. ";
      throw error(oss.str());
    }

    // Add all the stack contexts that the unprunenable tensor might appear in
    // to the set of unprunenable stack tensors.
    else {
      for (const auto &stack_ : found->second) {
        unpruneableStackIds.push_back({unpruneableTensorId, stack_});
      }
    }
  }

  // traverse backwards through the graph(s) to find all tensors which are on
  // a path to an unpruneable tensor.
  auto unpruneableBackClosure =
      querier.onMultiGraphPathTo(unpruneableStackIds);

  // An aside on tensors which alias each other:
  //
  // Some users might like to add all aliases of unpruneable tensors which are
  // modified to the set of unpruneable tensors. This could be done quite
  // easily: add them to the above closure, and then rerun the method with the
  // above closure as the new unrpuneableBackSources set. Repeat until no new
  // tensors added.
  //
  // However, a cleaner more SSA approach is to ensure that modifiers are
  // final consumers, in which case this is not needed.

  // Strip the stack information from the tensors.
  auto unpruneable = callstack::StackUtil::tensorIds(unpruneableBackClosure);
  return TensorIds{unpruneable.cbegin(), unpruneable.cend()};
}

void Pruner::prune(const callstack::Querier &gq,
                   Mutator &gm,
                   const SubGraphIds &callables,
                   const TensorIds &unpruneableBackSources) {

  // Within a graph, ops will be pruned from outputs to inputs, so that at the
  // time an Op is pruned (removed), none of its outputs is required by any
  // Op.
  //
  // The order in which the graphs will be pruned is top-down. That is the
  // "main" graphs first and the most nested callee graphs last. This is so
  // that at the time op is pruned (removed), all copies to/from its output
  // tensors have been removed already.
  //
  const auto opPruningOrder =
      gq.scheduled(callstack::Querier::DataDepOrder::Bwd,
                   callstack::Querier::GraphDepOrder::TopDown);

  auto ups = unpruneable(gq, callables, unpruneableBackSources);
  std::set<TensorId> unpruneableSet(ups.cbegin(), ups.cend());

  // Unpruneable ops are all ops which have at least 1 output tensor was
  // determined to be unpruneable.
  std::set<OpId> unpruneableOps;
  for (const auto &tId : unpruneableSet) {
    unpruneableOps.insert(tId.opId());
  }

  // Go through all ops, and prune as appropriate.
  for (auto opId : opPruningOrder) {

    // Op which is pruneable (all output tensors were pruneable.)
    if (unpruneableOps.count(opId) == 0) {
      for (OutIndex o = 0; o < gq.nOutTensors(opId); ++o) {
        if (gq.hasConsumers({opId, o})) {
          std::ostringstream oss;
          oss << "Cannot prune (remove) an op whose outputs have consumers. "
              << "This is the case for op " << opId
              << ", which has consumers " << gq.consumptionIds({opId, o})
              << " at output index " << o
              << ". Pruning was designed to proceed back graph outputs to "
              << "graph inputs, "
              << "so it is strange that there remains an output of " << opId
              << " which isn't already pruned. ";
          throw error(oss.str());
        }
      }

      const auto ctxt = "[pruning] " + gq.str(opId) +
                        " is not on a path to an unpruneable back source. ";

      // Remove the op. Things like topological constraint transfer must be
      // implemented by the user in this virtual method.
      gm.removeOp(opId, ctxt);
    }

    //  If the op is unpruneable (cannot be removed because it has an output
    //  on a path to an unpruneable back source), it might still be possible
    //  to remove some inputs and outputs if they are copies to callees.
    else if (gq.hasCallees(opId)) {

      // If the destination of a copy into a callee is pruneable, then the
      // copy can (and must) be removed. When would this happen? It could be
      // for example that the copy-in destionation is not on any path to a
      // copy-out source.
      InIndices insToRemove;
      for (auto p : gq.copyInDsts(opId)) {
        InIndex inIndex = p.first;
        TensorId tId    = p.second;
        if (unpruneableSet.count(tId) == 0) {
          insToRemove.push_back(inIndex);
        }
      }

      // If the destination of an out-copy is pruneable, then the copy-out can
      // be removed.
      OutIndices outsToRemove;
      for (OutIndex outIndex = 0; outIndex < gq.nOutTensors(opId);
           ++outIndex) {
        if (unpruneableSet.count({opId, outIndex}) == 0) {
          outsToRemove.push_back(outIndex);
        }
      }

      gm.removeInputs(opId, insToRemove);
      gm.removeOutputs(opId, outsToRemove);
    }
  }
}

} // namespace prune
} // namespace program
} // namespace poprithms
