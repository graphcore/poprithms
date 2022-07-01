// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/call.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

OptionalTensorIds CallDifferentiator::createInGrads(
    OpId callOpId,
    IAutomaticMutator &mutator,
    const IAutomaticQuerier &querier,
    const poprithms::autodiff::core::ToGradGraph &toGradGraph,
    const GradInfos &gradInfos,
    SubGraphId sgToExtend) {

  /*
   *  Illustration of the setup with variable names used in code.
   *
   *  g1 is the gradient of in1 (in callee sub-graphs)
   *  g0 is the gradient of in0 (in caller sub-graphs)
   *
   *  in0, in1, g0, g1 are the variable names in the code.
   *
   *                  forward graph:
   *   +---------------------------------------------+
   *   |                                             |
   *   |                  call op                    |
   *   |            +---------------------+          |
   *   |      in0-> | in1-+               | -> out0  |
   *   |            |     +-- ... --> ..  |          |
   *   |      ..->  |  ..-+               | -> ..    |
   *   |            +---------------------+          |
   *   |                                             |
   *   +---------------------------------------------+
   *
   *
   *                  backward graph:
   *   +----------------------------------------------------------+
   *   |                                                          |
   *   |            grad call op                                  |
   *   |          +---------------+                               |
   *   |   g0 <-- |  g1 <--- ..   |  <--  checkpoints             |
   *   |          |               |  <--     and                  |
   *   |   .. <-- |  .. <--- ..   |  <--  gradients (dOut0, etc.) |
   *   |          +---------------+        (copyIns)              |
   *   |                                                          |
   *   +----------------------------------------------------------+
   *
   *
   * This method creates g0.
   */

  // The graph where #g1 above lives. Note that it not in the remit of this
  // function to create this sub-graph.
  const SubGraphId callGradGraph = gradInfos.grad(callOpId, CalleeIndex(0));

  const auto &callGradInfo = gradInfos.at(callGradGraph);

  // The graph where #g0 above lives:
  const auto caller = sgToExtend;

  // The graph where #g1 above lives:
  const auto callee = callGradGraph;

  // The copies into the call op's gradient call. These consist of
  // 1) the checkpoints.
  // 2) the input gradients.
  std::vector<std::pair<TensorId, TensorId>> copyIns;

  const CallEvent fwdEvent{callOpId, querier.callee(callOpId, 0), 0};

  // 1) checkpoints.
  for (auto cpt : callGradInfo.checkpointPairs()) {
    auto dst = querier.dstInCaller(cpt.inNonGradGraph, fwdEvent);
    copyIns.push_back({toGradGraph.getNonGrad(dst), cpt.inGradGraph});
  }

  // 2) input gradients.
  for (auto x : callGradInfo.gradInPairs()) {
    auto dst = querier.dstInCaller(x.nonGradInNonGradGraph, fwdEvent);
    copyIns.push_back({toGradGraph.getGrad(dst), x.gradInGradGraph});
  }

  // gradients of some of the inputs (like g1 for in1) and possibly some
  // other gradient tensors if this graph is used in multiple cases.
  // We assume here that all the gradients we want are in this set.
  const auto targetGrads = callGradInfo.targetGradsInGradGraph();

  const auto bwdCall = mutator.call(caller, callee, copyIns, targetGrads);
  const CallEvent bwdEvent{bwdCall, querier.callee(bwdCall, 0), 0};

  const auto nFwdIns = querier.nInTensors(callOpId);

  // gradsOfIns will contain tensors like #g0.
  OptionalTensorIds gradsOfIns(nFwdIns);
  for (InIndex inIndex = 0; inIndex < nFwdIns; ++inIndex) {
    auto in1 = querier.inDst(callOpId, inIndex).tId();
    if (callGradInfo.objective().isTarget(in1)) {
      const auto g1             = callGradInfo.targetGradInGradGraph(in1);
      const auto g0             = querier.dstInCaller(g1, bwdEvent);
      gradsOfIns[inIndex.get()] = g0;
    }
  }

  return gradsOfIns;
}
} // namespace automatic
} // namespace autodiff
} // namespace poprithms
