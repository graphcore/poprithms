// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_ELEMENTWISE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_ELEMENTWISE_HPP

#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * An elementwise op with
 * - 2 inputs, which are numpy-broadcastable, and
 * - 1 output.
 * */
class BinaryElementwise : public WithoutCalleesTensorCentric {
public:
  BinaryElementwise(const Op::State &s) : WithoutCalleesTensorCentric(s) {}

  /**
   * This binary elementwise op modifies the input at index #i if it aliases
   * the input at index #i.
   * */
  bool modifies(InIndex i) const final { return aliases(i, 0); }

  /**
   * A binary elementwise op does computation, and is therefore not an
   * 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  /**
   * The output is not a reference to a tensor in another graph. (\sa the
   * RefFrom op class).
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }

  /**
   * Check that there are 2 inputs, 1 output, all inputs and outputs have the
   * same data type, etc. Once all the base class expectations have been
   * checked, check any derived class attributes.
   * */
  void computeDerivedVerifyValid() const final;

  /**
   * Check derived class attribute validity.
   * */
  virtual void binaryElementwiseDerivedVerifyValid() const = 0;

  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void runSim(SimTensorMap &htm) const final { runReplicatedSim(htm); }
};

/**
 * Non-aliasing binary elementwise op.
 *
 * Design note: why do have inplace and output versions? The view-changing ops
 * are only inplace, to do an 'outplace' view-change a copy must be inserted
 * manually. Why is this pattern not followed for elementwise ops too, it does
 * mean less ops and code? The main reason for this is that (1) backends might
 * provide different APIs for inplace and outplace and (2) SSA compute ops
 * might be useful. This might change though, we might remove outplace
 * elementwise ops.
 * */
class BinaryElementwiseOutplace : public BinaryElementwise {
public:
  BinaryElementwiseOutplace(const Op::State &s) : BinaryElementwise(s) {}

  bool aliases(InIndex, OutIndex) const final { return false; }

  /**
   * The output tensor is a new allocation, and so #inTensors is not used.
   * */
  HostTensors initializeOut(const HostTensors &inTensors) const final;

  /**
   * Create a new variable/allocation in the alias::Graph corresponding to the
   * output of this op.
   * */
  void growAliasMapper(MemoryAliasMapper &) const final;
};

class Add final : public BinaryElementwiseOutplace {
public:
  Add(const Op::State &s) : BinaryElementwiseOutplace(s) {}

  /**
   * Update the single tensor in #outs to be the sum of the 2 tensors in #ins.
   * */
  void compute(const HostTensors &ins, const HostTensors &outs) const final;

  /**
   * Sum-reduce the gradient of the output to each of the 2 input shapes.
   * */
  static OptionalTensors
  addBackpropagate(const GradOpIns &, const Shape &in0, const Shape &in1);
  OptionalTensors bprop(const GradOpIns &) const final;

  /**
   * The gradient is propagated from the output index to both of the input
   * indices.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }

  /**
   * The add op does not add any new attributes to its base class, so the
   * following methods are of the simplest form.
   * */
  std::string typeString() const final { return "Add"; }
  UpOp cloneWithState(const State &) const final;
  void binaryElementwiseDerivedVerifyValid() const final{};
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
