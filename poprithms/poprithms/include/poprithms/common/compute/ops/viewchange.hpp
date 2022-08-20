// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_VIEWCHANGE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_VIEWCHANGE_HPP

#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::util::Permutation;

/**
 * An op which does no computation, just presents a new view into the
 * input tensor(s).
 *
 * Note on the suffix '_' in the op name: this indicates that the op's output
 * aliases one or more of its inputs. We don't enforce this rule for naming
 * ops, although recommend that this rule of nomenclature be followed.
 *
 * Ops which inherit from the ViewChange_ op mostly correspond to the tensor
 * view-changes available in the poplar::Tensor API.
 * */
class ViewChange_ : public WithoutCalleesTensorCentric {
public:
  ViewChange_(const State &s) : WithoutCalleesTensorCentric(s) {}

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  bool isValueDependent(InIndex, OutIndex) const final { return true; }

  /**
   * This op does no computation, and is an 'initializing op' in this sense.
   * */
  bool isInitializingOp() const final { return true; }

  /**
   * As this op does no computation (it is an 'initializing op') it cannot
   * modify any of its input values.
   * */
  bool modifies(InIndex) const final { return false; }

  /**
   * The output tensor aliases all inputs.
   * */
  bool aliases(InIndex, OutIndex) const final { return true; }

  /**
   * This op does no computation.
   * */
  void runSim(ISimState &) const final {}

  /**
   * This op does no computation.
   * */
  void compute(const HostTensors &, const HostTensors &) const final {}

  /**
   * This op does no computation, so there is no code on ipu/host.
   * */
  CodeLocation codeLocation() const final { return CodeLocation::None; }

  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }

  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  /**
   * All view-changing ops propagate gradients through all input indices.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }

  /**
   * Only RefFrom ops can have root references which are not the output
   * tensors themselves.
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  void resetRootRef(OutIndex, const TensorId &) final;
};

/**
 * Permute the dimensions of a tensor.
 * */
class DimShuffle_ final : public ViewChange_ {
public:
  /**
   * \param p The permutation to apply to the input tensor.
   * */
  DimShuffle_(const State &s, const Permutation &p) : ViewChange_(s), p_(p) {}

  /**
   * The permutation to apply to the input tensor.
   * */
  const Permutation &permutation() const { return p_; }

  std::string typeString() const final;

  void growAliasMapper(MemoryAliasMapper &) const final;

  /**
   * Verify that the Permutation of this op has the correct rank for the input
   * tensor.
   * */
  void computeDerivedVerifyValid() const final;

  /**
   * Perform an aliasing dimShuffle on the unique tensor in #ins using this
   * op's Permutation.
   * */
  HostTensors initializeOut(const HostTensors &ins) const final;

  /**
   * \return true if this op is an identity view-change. That is, if the input
   *        and output have the same shape and the (row-major) order of the
   *        elements is unchanged.
   * */
  static bool isIdentity(const Shape &inShape,
                         const Shape &outShape,
                         const Permutation &);

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  UpOp cloneWithState(const State &) const final;

  /**
   * The gradient of a dimension shuffle operation is the inverse dimension
   * shuffle.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;

private:
  /**
   * \return true if the DimShuffle_ op #rhs (this method is only called when
   *         #rhs is a DimShuffle_) has the same permutation.
   * */
  bool computeTypeSpecificEqualTo(const Op &rhs) const final;

  Permutation p_;
};

/**
 * Reshape a tensor.
 * */
class Reshape_ final : public ViewChange_ {
public:
  Reshape_(const State &s) : ViewChange_(s) {}

  std::string typeString() const final { return "Reshape_"; }

  /**
   * Initialize the output tensor to be an alias of the input (the first and
   * only element of #ins).
   * */
  HostTensors initializeOut(const HostTensors &ins) const final;

  void computeDerivedVerifyValid() const final;

  OptionalTensors bprop(const GradOpIns &) const final;

  std::unique_ptr<Op> cloneWithState(const State &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void growAliasMapper(MemoryAliasMapper &b) const final;

  static bool isIdentity(const Shape &inShape, const Shape &outShape);

private:
  /**
   * As this op class has no additional attributes, any reshape op which is
   * equivalent to it at the base op level (common::compute::Op) will always
   * be equivalent to it overall.
   * */
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Reverse a tensor along one or several dimensions.
 * */
class Reverse_ final : public ViewChange_ {

public:
  Reverse_(const State &, const Dimensions &);

  /**
   * The dimensions to reverse.
   * */
  const Dimensions &dimensions() const { return dimensions_; }

  std::string typeString() const final;

  void computeDerivedVerifyValid() const final;

  /**
   * Insert a reverse op into the alias::Graph of #mam.
   * */
  void growAliasMapper(MemoryAliasMapper &mam) const final;

  HostTensors initializeOut(const HostTensors &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  /**
   * \return true if the reverse dimensions in #revDims are all in singleton
   *         dimensions of the input shape #inShape.
   * */
  static bool isIdentity(const Shape &inShape,
                         const Shape &outShape,
                         const Dimensions &revDims);

  std::unique_ptr<Op> cloneWithState(const State &) const final;

  /**
   * The gradient of a reversal operation is the same reversal operation.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;

private:
  bool computeTypeSpecificEqualTo(const Op &) const final;

  Dimensions dimensions_;
};

/**
 * Exand a tensor. This is a broadcasting view-change.
 * */
class Expand_ final : public ViewChange_ {
public:
  Expand_(const State &s);

  void computeDerivedVerifyValid() const final;

  /**
   * This op has no additional attributes, so the string for it is simply the
   * name of the op.
   * */
  std::string typeString() const final { return "Expand_"; }

  HostTensors initializeOut(const HostTensors &) const final;

  UpOp cloneWithState(const State &s) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void growAliasMapper(MemoryAliasMapper &b) const final;

  /**
   * Perform a sum-reduction of the gradient of the output, reducing to the
   * shape of the input (un-expanded) tensor.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;

  /**
   * An expand op is an identity view-change if and only if the input and
   * output shapes are identical.
   * */
  static bool isIdentity(const Shape &i, const Shape &o) { return i == o; }

private:
  // This op introduces no additional attributes, so this method always
  // returns true (it is only called when the base classes have been
  // established to be equivalent).
  bool computeTypeSpecificEqualTo(const Op &) const final;
};

/**
 * Statically slice a tensor.
 * */
class Slice_ final : public ViewChange_ {
public:
  /**
   * \param lower  The lower bounds of the slice. The rank of this vector must
   *               be the same as the input tensor's.
   *
   * \param upper  The upper bounds of the slice. The rank of this vector must
   *               be the same as the input tensor's.
   *
   * The returned tensor will have shape 'lower - upper'. That is, all
   * elements between the bounds #lower and #upper are retained.
   * */
  Slice_(const State &s, const Lower &lower, const Upper &upper);

  /**
   * The lower bounds of the slice.
   * */
  const Lower &lower() const { return lower_; }

  std::vector<uint64_t> lower_u64() const {
    return {lower_.cbegin(), lower_.cend()};
  }

  /**
   * The upper bounds of the slice.
   * */
  const Upper &upper() const { return upper_; }

  std::vector<uint64_t> upper_u64() const {
    return {upper_.cbegin(), upper_.cend()};
  }

private:
  /**
   * \return true if #sliceOp has the same lower and upper bounds. Note that
   *         this method is only called when it is known that #sliceOp is
   *         of type Slice.
   * */
  bool computeTypeSpecificEqualTo(const Op &sliceOp) const final;

  std::string typeString() const final;

  void computeDerivedVerifyValid() const;

  HostTensors initializeOut(const HostTensors &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  /**
   * Backpropagation of the slice op. The current implementation pads the
   * sliced tensor with a broadcast zero constant, back up to the shape of the
   * slice's input. This might need changing if it required that the gradient
   * is non-constant.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;

  void growAliasMapper(MemoryAliasMapper &b) const final;

  std::unique_ptr<Op> cloneWithState(const State &s) const final;

private:
  Lower lower_;
  Upper upper_;
};

/**
 * Concatenate tensors together along a specified dimension.
 * */
class Concat_ final : public ViewChange_ {
public:
  Concat_(const State &s, uint64_t axis)
      : ViewChange_(s), axis_(axis),
        partitionPoints_(Shape::concatPartitionPoints(s.inShapes(), axis)) {}

  /**
   * \return The axis of concatenation.
   * */
  uint64_t axis() const { return axis_; }

  /**
   * \return To slice the input at index #i out of the concatenated tensor,
   *         these are the lower bounds to use.
   * */
  std::vector<int64_t> lowerSlice(InIndex) const;

  /**
   * \return To slice the input at index #i out of the concatenated tensor,
   *         these are the upper bounds to use.
   * */
  std::vector<int64_t> upperSlice(InIndex) const;

  /**
   * Given a tensor #toSlice which is the same shape as the output of this op,
   * slice it into tensors which are the same shapes as this op's input
   * tensors.
   * */
  Tensors slice_(const Tensor &toSlice) const;

private:
  /**
   * Slice the gradient of the output into tensors with identical shapes to
   * this ops inputs.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;

  void computeDerivedVerifyValid() const final;

  void growAliasMapper(MemoryAliasMapper &b) const final;

  std::unique_ptr<Op> cloneWithState(const State &) const final;
  std::string typeString() const final;
  bool computeTypeSpecificEqualTo(const compute::Op &rhs) const final;

  HostTensors initializeOut(const HostTensors &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  uint64_t axis_;

  // the indices along the axis of concatenation where the concatenated
  // Tensors touch.
  std::vector<int64_t> partitionPoints_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
