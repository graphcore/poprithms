// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_VIEWCHANGE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_VIEWCHANGE_HPP

#include <poprithms/common/compute/ops/withoutcallees.hpp>
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
class ViewChange_ : public WithoutCallees {
public:
  ViewChange_(const Op::State &s) : WithoutCallees(s) {}

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

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
  void runSim(SimTensorMap &) const final {}

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
   * Map the regions #regs forward through this op. There is one Region per
   * input, so the size of regs is the same as the number of inputs of this
   * op.
   *
   * \sa poprithms::nest::Region
   * */
  virtual DisjointRegions apply(const std::vector<Region> &regs) const = 0;

  /**
   * Only RefFrom ops can have root references which are not the output
   * tensors themselves.
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
};

/**
 * A view-change op with a single input (currently all view-changing ops
 * except for Concat which has a variadic number of inputs).
 * */
class UnaryViewChange_ : public ViewChange_ {
public:
  UnaryViewChange_(const State &s) : ViewChange_(s) {}

  /**
   * As there is only 1 input to this op, #regions will always be a vector
   * with just 1 Region. The logic of mapping 1 input region to an output
   * region is implemented in #applyTo, which this method calls.
   * */
  DisjointRegions apply(const std::vector<Region> &regions) const final;

  /**
   * Map the input region #reg to an output region.
   * */
  virtual DisjointRegions applyTo(const Region &reg) const = 0;
};

/**
 * Permute the dimensions of a tensor.
 * */
class DimShuffle_ final : public UnaryViewChange_ {
public:
  /**
   * \param p The permutation to apply to the input tensor.
   * */
  DimShuffle_(const State &s, const Permutation &p)
      : UnaryViewChange_(s), p_(p) {}

  /**
   * The permutation to apply to the input tensor.
   * */
  const Permutation &permutation() const { return p_; }

  std::string typeString() const final;

  DisjointRegions applyTo(const Region &) const final;

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
   * \return true of this op is an identity view-change. That is, if the input
   *        and output have the same shape and the (row-major) order of the
   *        elements is unchanged.
   * */
  static bool isIdentity(const Shape &inShape,
                         const Shape &outShape,
                         const Permutation &);

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  UpOp cloneWithState(const Op::State &) const final;

  /**
   * The gradient of a dimension shuffle operation is the inverse dimension
   * shuffle.
   * */
  OptionalTensorIds backpropagate(Graph &, const GradOpInIds &) const final;

private:
  /**
   * \return true of the DimShuffle_ op #rhs (this method is only called when
   *         #rhs is a DimShuffle_) has the same permutation.
   * */
  bool computeTypeSpecificEqualTo(const Op &rhs) const final;

  Permutation p_;
};

/**
 * Reshape a tensor.
 * */
class Reshape_ final : public UnaryViewChange_ {
public:
  Reshape_(const Op::State &s) : UnaryViewChange_(s) {}

  std::string typeString() const final { return "Reshape_"; }

  /**
   * Initialize the output tensor to be an alias of the input (the first and
   * only elements if #ins).
   * */
  HostTensors initializeOut(const HostTensors &ins) const final;

  void computeDerivedVerifyValid() const final;

  DisjointRegions applyTo(const Region &) const final;

  OptionalTensorIds backpropagate(Graph &g, const GradOpInIds &) const final;

  std::unique_ptr<Op> cloneWithState(const Op::State &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void growAliasMapper(MemoryAliasMapper &b) const final;

  static bool isIdentity(const Shape &inShape, const Shape &outShape);

private:
  /**
   * As this op class has no additional attributes, any reshape op which is
   * equivalent to it at the base op level (compute::Op) will always be
   * equivalent to it overall.
   * */
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * Reverse a tensor along one or several dimensions.
 * */
class Reverse_ final : public UnaryViewChange_ {

public:
  Reverse_(const Op::State &, const Dimensions &);

  /**
   * The dimensions to reverse.
   * */
  const Dimensions &dimensions() const { return dimensions_; }

  std::string typeString() const final;

  void computeDerivedVerifyValid() const final;

  /**
   * Reverse the dimensions of the Region #r.
   * */
  DisjointRegions applyTo(const Region &r) const final;

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

  std::unique_ptr<Op> cloneWithState(const Op::State &) const final;

  /**
   * The gradient of a reversal operation is the same reversal operation.
   * */
  OptionalTensorIds backpropagate(Graph &, const GradOpInIds &) const final;

private:
  bool computeTypeSpecificEqualTo(const Op &) const final;

  Dimensions dimensions_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
