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
  void resetRootRef(OutIndex, const TensorId &) final { invalid(); }

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

/**
 * Aliasing (inplace) binary elementwise op.
 *
 * The output is an alias of the input at index #0.
 * */
class BinaryElementwiseInplace_ : public BinaryElementwise {
public:
  /**
   * The output is aliased to the input #0.
   * */
  bool aliases(InIndex i, OutIndex) const final { return i == 0; }

  BinaryElementwiseInplace_(const Op::State &s) : BinaryElementwise(s) {}

  /**
   * The output is an alias of the input at index 0.
   * */
  void growAliasMapper(MemoryAliasMapper &) const final;
  HostTensors initializeOut(const HostTensors &ins) const final;

  /**
   * Check that the first input numpy-dominates the second input. Then,
   * checked derived class validity.
   * */
  void binaryElementwiseDerivedVerifyValid() const final;

protected:
  virtual void binaryElementwiseInplaceDerivedVerifyValid() const = 0;

  // Used by ops which do not have autodiff.
  std::string noInplaceAutodiff() const;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
