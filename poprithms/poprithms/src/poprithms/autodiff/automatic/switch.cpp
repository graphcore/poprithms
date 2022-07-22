// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/switch.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

/**
 * Example 1
 * ---------
 *
 * forwards: 2 inputs and 1 output, 2 sub-graphs.
 *
 * if cond is 0: inA gets copied to in0 in sub-graph 0, and sub-graph 0 runs.
 * if cond is 1: inB gets copied to in1 in sub-graph 1, and sub-graph 1 runs.
 *
 *         +----------------------------+
 *         |       sub-graph 0          |
 * inA ->  | in0 ---------------> out0  |
 *         |                            |          (value of out is
 *         |                            +--> out    out0 or out1,
 *         |                            |           depending on which
 * inB ->  | in1----------------> out1  |           path was taken)
 *         |       sub-graph 1          |
 * cond -> |                            |
 *         +----------------------------+
 *
 * backwards: the outputs are gradients of the forward graph inputs.
 *
 *               +------------------------------+
 *               |     grad sub-graph 0         |
 *   dInA <----  | din0 ------------<--- out0   +---<--- cond (same as in
 *               |                              |              forward op).
 *               |                              +--<-- dOut
 *               |                              |
 *   dInB <----  | din1--------------<-- dout1  +---<-- checkpoints from
 *               |     grad sub-graph 1         |     both forward sub-graphs
 *               |                              |
 *               +------------------------------+
 *
 * dInA is the gradient of inA and dInB is the gradient of inB.
 *
 * If cond is 0, then dInB will be zero.
 * If cond is 1, then dInA will be zero.
 *
 *
 *
 * Example 2
 * ---------
 * Suppose that in the above example, inA = inB. That is, inA is copied to in0
 * if cond is 0, and inA is copied to in1 if cond is 1:
 *
 *           +----------------------------+
 *           |       sub-graph 0          |
 *       +-> | in0 ---------------> out0  |
 *       |   |                            |
 * inA --+   |                            +--> out
 *       |   |                            |
 *       +-> | in1----------------> out1  |
 *           |       sub-graph 1          |
 *           |                            |
 *           +----------------------------+
 *
 * In this case, the backwards graph is identical to in example 1, and has 2
 * outputs, one for each of the copies from inA. The gradient of inA is the
 * sum of these 2 gradient outputs, one of which is always zero. There is a
 * task TODO(T66529) to elide this add-0.
 *
 * */

OptionalTensorIds
SwitchDifferentiator::createInGrads(OpId swOpId,
                                    IAutomaticMutator &gm,
                                    const IAutomaticQuerier &gq,
                                    const core::ToGradGraph &toGradGraph,
                                    const GradInfos &gradInfos,
                                    SubGraphId toExtend,
                                    const TensorId &conditionId) {

  // For each callee, the gradient callee.
  SubGraphIds gradCallees;
  for (CalleeIndex ci = 0; ci < gq.nCallees(swOpId); ++ci) {
    gradCallees.push_back(gradInfos.grad(swOpId, ci));
  }

  // The copies into the switch op's gradient switch. These consist of
  // 1) the checkpoints.
  // 2) the input gradients.
  std::vector<std::tuple<TensorId, TensorId, CalleeIndex>> copyIns;

  for (CalleeIndex ci = 0; ci < gq.nCallees(swOpId); ++ci) {
    const auto callGradGraph = gradInfos.grad(swOpId, ci);
    const auto &gradInfo_    = gradInfos.at(callGradGraph);
    CallEvent fwdEvent(swOpId, gq.callee(swOpId, ci), ci);

    // 1) Checkpoints.
    for (auto cpt : gradInfo_.checkpointPairs()) {
      auto dst = gq.dstInCaller(cpt.inNonGradGraph, fwdEvent);
      copyIns.push_back({toGradGraph.getNonGrad(dst), cpt.inGradGraph, ci});
    }

    // 2) Input gradients.
    for (auto x : gradInfo_.gradInPairs()) {
      auto dst = gq.dstInCaller(x.nonGradInNonGradGraph, fwdEvent);
      copyIns.push_back({toGradGraph.getGrad(dst), x.gradInGradGraph, ci});
    }
  }

  // For each of the (non-condition) inputs: Is there are gradient provided?
  std::vector<bool> containsGrad(gq.nInCopies(swOpId), false);

  // A vector of vectors which is of shape [nInTensors][nCallees], this
  // contains many zero tensors. There is task to simplify this sparse
  // structure TODO(T66529).
  std::vector<TensorIds> gradOuts;
  gradOuts.reserve(gq.nInCopies(swOpId));
  for (InIndex i = 0; i < gq.nInCopies(swOpId); ++i) {

    TensorIds outsAtIndex;
    for (CalleeIndex ci = 0; ci < gq.nCallees(swOpId); ++ci) {

      const auto callGradGraph = gradInfos.grad(swOpId, ci);
      const auto &gradInfo_    = gradInfos.at(callGradGraph);

      auto t = gq.inDst(swOpId, i).tId();
      if (gq.inDst(swOpId, i).calleeIndex() == ci &&
          gradInfo_.objective().isTarget(t)) {
        outsAtIndex.push_back(gradInfo_.targetGradInGradGraph(t));
        containsGrad.at(i.get()) = true;
      } else {
        SubGraphId sgId = gradInfos.grad(swOpId, ci);
        std::ostringstream oss;
        oss << "switch-grad-zero-" << i << ":" << ci;
        auto z = gm.zeroLike(t, sgId, oss.str());
        outsAtIndex.push_back(z);
      }
    }
    gradOuts.push_back(outsAtIndex);
  }

  const auto bwdSwitch = gm.switchOp(toExtend,
                                     gradCallees,
                                     toGradGraph.getNonGrad(conditionId),
                                     copyIns,
                                     gradOuts,
                                     {});

  OptionalTensorIds gradsOfIns(gq.nInTensors(swOpId));
  for (InIndex inIndex = 0; inIndex < gq.nInCopies(swOpId); ++inIndex) {
    if (containsGrad.at(inIndex.get())) {
      gradsOfIns.at(inIndex.get()) = TensorId(bwdSwitch, inIndex.get());
    }
  }

  return gradsOfIns;
}

} // namespace automatic
} // namespace autodiff
} // namespace poprithms
