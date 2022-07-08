// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_NOP_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_NOP_HPP

#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * A Nop ("no op") is an op which does no computation, has no input tensors,
 * and no output tensors.
 *
 * One use of this op is as a barrier to separate groups of ops when
 * scheduling a graph (see Graph::insertBinBoundary).
 * */
class Nop final : public WithoutCallees {
public:
  Nop(const Op::State &s) : WithoutCallees(s) {}
  CodeLocation codeLocation() const final { return CodeLocation::None; }

  /**
   * This op has no inputs and no outputs, so any input or output index passes
   * as an argument to any of the following methods must be invalid, and if
   * called an error is thrown.
   * */
  bool aliases(InIndex, OutIndex) const final { invalid(); }
  bool modifies(InIndex) const final { invalid(); }
  TensorId rootRef(OutIndex) const final { invalid(); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
  bool gradientPropagates(OutIndex, InIndex) const final { invalid(); }

  /** There is no computation to run for a no-op at runtime. */
  bool isInitializingOp() const final { return true; }
  void runSim(ISimState &) const final {}
  void compute(const HostTensors &, const HostTensors &) const final{};

  /**
   * This op has no outputs, so there is are no outputs to initialize.
   * */
  void initializeSimOut(SimTensorMap &) const final{};
  HostTensors initializeOut(const HostTensors &) const final { return {}; }
  void growAliasMapper(MemoryAliasMapper &) const final {}

  /** Nop has no additional attributes, so all Nops are the same. */
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /** Assert that there are no inputs and no outputs.  */
  void computeDerivedVerifyValid() const final {
    OpVerifier(*this).verifyNonVariadicFromAtts(0, 0, {});
  }

  OptionalTensorIds backpropagate(Graph &, const GradOpInIds &) const final {
    return {};
  }

  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  std::string typeString() const final { return "Nop"; }
  std::unique_ptr<Op> cloneWithState(const State &) const final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
