// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_BINARYELEMENTWISE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_BINARYELEMENTWISE_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/ops/withautodiff.hpp>

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
  BinaryElementwise(const State &s) : WithoutCalleesTensorCentric(s) {}

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
 * */
class BinaryElementwiseOutplace : public BinaryElementwise {
public:
  BinaryElementwiseOutplace(const State &s) : BinaryElementwise(s) {}

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

protected:
  /**
   * Check that there are 2 inputs, 1 output, all inputs and outputs have the
   * same data type, etc. Used for the 'standard' ops (Add, Mul, etc.)
   * */
  void simpleBinaryElementwiseOutplaceVerifyValid() const;
};

/**
 * Aliasing (inplace) binary elementwise op.
 *
 * The output is an alias of the input at index 0.
 * */
class BinaryElementwiseInplace_ : public BinaryElementwise {
public:
  /**
   * The output is aliased to the input 0.
   * */
  bool aliases(InIndex i, OutIndex) const final { return i == 0; }

  BinaryElementwiseInplace_(const State &s) : BinaryElementwise(s) {}

  /**
   * The output is an alias of the input at index 0.
   * */
  void growAliasMapper(MemoryAliasMapper &) const final;
  HostTensors initializeOut(const HostTensors &ins) const final;

protected:
  // Used by ops which do not have autodiff.
  [[noreturn]] void noInplaceAutodiff() const;

  /**
   * Check that there are 2 inputs, 1 output, all inputs and outputs have the
   * same data type, the first input numpy-dominates the second.
   * */
  void simpleBinaryElementwiseInplaceVerifyValid() const;
};

class Add final : public WithAutodiff<autodiff::automatic::AddAutodiffer,
                                      BinaryElementwiseOutplace> {

public:
  Add(const State &s) : WithAutodiff(s) {}

private:
  /**
   * Update the single tensor in #outs to be the sum of the 2 tensors in #ins.
   * */
  void compute(const HostTensors &ins, const HostTensors &outs) const final;

  /**
   * The add op does not add any new attributes to its base class, so the
   * following methods are of the simplest form.
   * */
  std::string typeString() const final { return "Add"; }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  UpOp cloneWithState(const State &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }
};

/**
 * Add the input at index 1 to the input at index 0, inplace.
 *
 * This op, even though it is inplace, can propagate the output gradient to
 * the 2 inputs, because neither of the inputs are used, and so it doesn't
 * matter that the first input has had its value changed.
 * */
class Add_ final : public WithAutodiff<autodiff::automatic::AddAutodiffer,
                                       BinaryElementwiseInplace_> {

public:
  Add_(const State &s) : WithAutodiff(s) {}

private:
  using AD = autodiff::automatic::AddAutodiffer;
  UpOp cloneWithState(const State &) const final;

  std::string typeString() const final { return "Add_"; }
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  void compute(const HostTensors &ins, const HostTensors &outs) const final {
    outs[0].add_(ins[1]);
  }

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }
};

/**
 * Multiply 2 tensors together.
 * */
class Mul final : public WithAutodiff<autodiff::automatic::MulAutodiffer,
                                      BinaryElementwiseOutplace> {
public:
  Mul(const State &s) : WithAutodiff(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Mul"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Multiply 2 tensors together, inplace on the first tensor.
 * */
class Mul_ final : public BinaryElementwiseInplace_ {
public:
  Mul_(const State &s) : BinaryElementwiseInplace_(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Mul_"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /**
   * This inplace op cannot be differentiated, as input values are required
   * for the backwards op but the first input value is written to inplace by
   * this op.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final {
    noInplaceAutodiff();
  }
  OptionalTensors bprop(const GradOpIns &) const final {
    noInplaceAutodiff();
  }
  InIndices autodiffRequiredIns() const final { noInplaceAutodiff(); }
  OutIndices autodiffRequiredOuts() const final { noInplaceAutodiff(); }
};

/**
 * Divide one tensor by another, inplace. This op is the operator '/=' with
 * numpy broadcasting.
 *
 * This inplace op does support automatic differentiation: to compute the
 * gradients of the inputs does not require the value of the modified input
 * (the numerator). The gradients of the numerator and denominator can be
 * computed with just the output (quotient) and the denominator, values which
 * are available after the division has been performed.
 *
 * \sa DivAutodiffer.
 * */

class Div_ final : public WithAutodiff<autodiff::automatic::DivAutodiffer,
                                       BinaryElementwiseInplace_> {
public:
  Div_(const State &s) : WithAutodiff(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Div_"; }

  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Divide one tensor by another.
 * */
class Div final : public WithAutodiff<autodiff::automatic::DivAutodiffer,
                                      BinaryElementwiseOutplace> {
public:
  Div(const State &s) : WithAutodiff(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Div"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * First tensor to the power of the second tensor.
 * */
class Pow final : public WithAutodiff<autodiff::automatic::PowAutodiffer,
                                      BinaryElementwiseOutplace> {
public:
  Pow(const State &s) : WithAutodiff(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Pow"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

class Pow_ final : public BinaryElementwiseInplace_ {
public:
  Pow_(const State &s) : BinaryElementwiseInplace_(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Pow_"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /**
   * There is not differentiation for this inplace op, as both inputs are
   * required.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;
  bool gradientPropagates(OutIndex, InIndex) const final;
  std::vector<InIndex> autodiffRequiredIns() const final;
  std::vector<OutIndex> autodiffRequiredOuts() const final;
};

/**
 * Binary operation which compares 2 tensors, with numpy broadcasting. The
 * output tensor is a boolean tensor.
 * */
class GreaterThan final : public BinaryElementwiseOutplace {
public:
  GreaterThan(const State &s) : BinaryElementwiseOutplace(s) {}

private:
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "GreaterThan"; }
  void compute(const HostTensors &, const HostTensors &) const final;
  void computeDerivedVerifyValid() const final;

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /**
   * As the output is boolean, this op is never differentiable.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final { return false; }
  OptionalTensors bprop(const GradOpIns &) const final {
    boolReturnAutodiff();
  }
  std::vector<InIndex> autodiffRequiredIns() const final {
    boolReturnAutodiff();
  }
  std::vector<OutIndex> autodiffRequiredOuts() const final {
    boolReturnAutodiff();
  }

  [[noreturn]] void boolReturnAutodiff() const;
};

/**
 * Subtraction operation.
 * */
class Sub final : public WithAutodiff<autodiff::automatic::SubAutodiffer,
                                      BinaryElementwiseOutplace> {
public:
  Sub(const State &s) : WithAutodiff(s) {}
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Sub"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Subtraction operation, inplace. Note that autodiff is supported, as neither
 * of the inputs to the op are required in the backwards pass.
 * */
class Sub_ final : public WithAutodiff<autodiff::automatic::SubAutodiffer,
                                       BinaryElementwiseInplace_> {
public:
  Sub_(const State &s) : WithAutodiff(s) {}
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Sub_"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Remainder operation.  Example:
 *
 * input 0 : [1 2 3 4]
 * input 1 : [3]
 * output  : [1 2 0 1]
 *
 * The Remainder op has no attributes. It can have floating point inputs. It
 * always propagates a zero gradient to its input.
 *
 * This is identical to C++ fmod.
 * */
class Remainder final
    : public Attributeless<ZeroAutodiff<BinaryElementwiseOutplace>,
                           Remainder> {
public:
  Remainder(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Remainder";
  void compute(const HostTensors &ins, const HostTensors &outs) const final {
    outs[0].update_(ins[0].mod(ins[1]));
  }

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }
};

/**
 * Remainder operation, inplace.
 * */
class Remainder_ final
    : public Attributeless<ZeroAutodiff<BinaryElementwiseInplace_>,
                           Remainder_> {

public:
  Remainder_(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Remainder_";

  void compute(const HostTensors &ins, const HostTensors &) const final {
    ins[0].mod_(ins[1]);
  }

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }
};

static const constexpr int CopyFromSourceIndex{1};

/**
 * Copy the values from one tensor to another.
 * */
class CopyFrom_ final
    : public WithAutodiff<
          autodiff::automatic::CopyAutodiffer<CopyFromSourceIndex>,
          BinaryElementwiseInplace_> {
public:
  CopyFrom_(const State &s) : WithAutodiff(s) {}
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "CopyFrom_"; }
  /**
   * The input index of the tensor which is updated (copied to).
   * */
  static InIndex Destination() { return 0; }
  TensorId destinationId() const { return inTensorId(Destination()); }

  /**
   * The input index od the tensor which is copied from.
   * */
  static InIndex Source() { return CopyFromSourceIndex; }
  TensorId sourceId() const { return inTensorId(Source()); }

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  void compute(const HostTensors &, const HostTensors &) const final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
