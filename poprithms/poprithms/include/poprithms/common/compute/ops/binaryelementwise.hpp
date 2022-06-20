// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_BINARYELEMENTWISE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_BINARYELEMENTWISE_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OutIndices;

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

/**
 * An op class which leverages the automatic differentiation methods of
 * another class, AD, to define the autodiff specific virtual methods.
 *
 * \tparam AD the class which defines the autodiff methods.
 * */
template <class AD>
class BinaryElementwiseOutplaceWithAutodiffer
    : public BinaryElementwiseOutplace {
public:
  BinaryElementwiseOutplaceWithAutodiffer(const State &s)
      : BinaryElementwiseOutplace(s) {}

  InIndices autodiffRequiredIns() const final {
    return AD::autodiffRequiredIns();
  }

  OutIndices autodiffRequiredOuts() const final {
    return AD::autodiffRequiredOuts();
  }

  OptionalTensors bprop(const GradOpIns &g) const final {
    return AD::backpropagate(g);
  }

  bool gradientPropagates(OutIndex o, InIndex i) const final {
    return AD::gradientPropagates(o, i);
  }
};

class Add final : public BinaryElementwiseOutplace {

public:
  Add(const State &s) : BinaryElementwiseOutplace(s) {}

private:
  using AD = poprithms::autodiff::automatic::AddAutodiffer;

  /**
   * Update the single tensor in #outs to be the sum of the 2 tensors in #ins.
   * */
  void compute(const HostTensors &ins, const HostTensors &outs) const final;

  bool gradientPropagates(OutIndex o, InIndex i) const final {
    return AD::gradientPropagates(o, i);
  }
  InIndices autodiffRequiredIns() const final {
    return AD::autodiffRequiredIns();
  }
  OutIndices autodiffRequiredOuts() const final {
    return AD::autodiffRequiredOuts();
  };
  OptionalTensors bprop(const GradOpIns &gIn) const final {
    return AD::backpropagate(gIn, inShape(0), inShape(1));
  }

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
 * */
class Add_ final : public BinaryElementwiseInplace_ {

public:
  Add_(const State &s) : BinaryElementwiseInplace_(s) {}

private:
  using AD = poprithms::autodiff::automatic::AddAutodiffer;
  UpOp cloneWithState(const State &) const final;

  std::string typeString() const final { return "Add_"; }
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /**
   * This op, even though it is inplace, can propagate the output gradient to
   * the 2 inputs, because neither of the inputs are used and so it doesn't
   * matter that the first input has been overriden.
   * */
  bool gradientPropagates(OutIndex o, InIndex i) const final {
    return AD::gradientPropagates(o, i);
  }
  InIndices autodiffRequiredIns() const final {
    return AD::autodiffRequiredIns();
  }
  OutIndices autodiffRequiredOuts() const final {
    return AD::autodiffRequiredOuts();
  };
  OptionalTensors bprop(const GradOpIns &gIn) const final {
    return AD::backpropagate(gIn, inShape(0), inShape(1));
  }

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
class Mul final : public BinaryElementwiseOutplaceWithAutodiffer<
                      poprithms::autodiff::automatic::MulAutodiffer> {
public:
  Mul(const State &s) : BinaryElementwiseOutplaceWithAutodiffer(s) {}

private:
  std::unique_ptr<Op> cloneWithState(const State &) const final;
  std::string typeString() const final { return "Mul"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseOutplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Multiply 2 tensors together, setting the the value of the first input to
 * the computed product.
 * */
class Mul_ final : public BinaryElementwiseInplace_ {
public:
  Mul_(const State &s) : BinaryElementwiseInplace_(s) {}

private:
  std::unique_ptr<Op> cloneWithState(const State &) const final;
  std::string typeString() const final { return "Mul_"; }
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    simpleBinaryElementwiseInplaceVerifyValid();
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /**
   * This inplace op cannot be differentiated.
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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
