// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_SWITCH_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_SWITCH_HPP

#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

/**
 * Utility class for differentiating a switch op.
 * */
class SwitchDifferentiator {

public:
  /**
   * Create gradients for the inputs of the op #switchOpId, using the
   * gradients of the outputs and the checkpointed callee tensors.
   *
   * The created gradient op, which is itself a switch op, will be inserted
   * into the graph #toExtend.
   *
   * The gradient switch op has 1 output for every input of the forward switch
   * op, other than the conditional tensor (#conditionId).
   *
   * As an example, suppose that the switch op has 3 sub-graphs (so
   * #conditionId is always 0, 1, or 2), and that for each of the 3 paths
   * through the switch there is 1 copy, always from the same tensor in the
   * calling scope):
   *
   *           +----------------------------+
   *           |                            |
   *       +---| in0 -> sub-graph 0 -> out0 |
   *       |   |                            |
   * in >--+---| in1 -> sub-graph 1 -> out1 +----> out
   *       |   |                            |
   *       +---| in2 -> sub-graph 2 -> out2 |
   *           |                            |
   * cond -----+                            |
   *           +----------------------------+
   *
   * Then the gradient switch op will have 3 outputs, 2 of which will be zero:
   *
   *            +--------------------------------+
   *            |                                |
   *        <---| dIn0 < grad sub-graph 0 <--    |
   *            |                                |
   *        <---| dIn1 < grad sub-graph 1 <--    |<---- dOut
   *            |                                |
   *        <---| dIn2 < grad sub-graph 2 <--    |<---- checkpoints
   *            |                                |
   *            |                                |<---- cond
   *            +--------------------------------+
   *
   * For example, if cond is 1 then zero tensors are copied out for dIn0 and
   * dIn2.
   * */
  static OptionalTensorIds createInGrads(OpId switchOpId,
                                         IAutomaticMutator &gm,
                                         const IAutomaticQuerier &gq,
                                         const core::ToGradGraph &toGradGraph,
                                         const GradInfos &gradInfos,
                                         SubGraphId toExtend,
                                         const TensorId &conditionId);
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
