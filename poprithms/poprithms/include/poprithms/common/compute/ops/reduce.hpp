// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_REDUCE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_REDUCE_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/gradopins.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withautodiff.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/util/stridedpartition.hpp>

namespace poprithms {
namespace common {
namespace compute {

using util::StridedPartition;

/**
 * Reduce a tensor along a subset of its dimensions. The reduced (output)
 * tensor has the same rank as the original tensor, i.e. singleton dimensions
 * are not removed ('squeezed').
 * */
class Reduce : public WithoutCalleesTensorCentric {

public:
  Reduce(const State &s, const Dimensions &dims)
      : WithoutCalleesTensorCentric(s), dims_(dims) {}

  /**
   * Reduce the tensor along all dimensions, so that the resulting tensor has
   * just 1 element, while retaining the same rank as the input tensor.
   * */
  Reduce(const State &s)
      : Reduce(s, /* reduction dimensions = */ s.inShape(0).dimensions()) {}

  /**
   * The dimensions that are reduced to singletons.
   * */
  const Dimensions &dimensions() const { return dims_; }

  /**
   * The output is not a reference to a tensor in another graph. (\sa the
   * RefFrom op class).
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }

  bool computeTypeSpecificEqualTo(const Op &) const final;

  /**
   * The input does not alias the input.
   * */
  bool aliases(InIndex, OutIndex) const final { return false; }
  bool modifies(InIndex) const final { return false; }
  HostTensors initializeOut(const HostTensors &) const final;
  void growAliasMapper(MemoryAliasMapper &b) const final {
    createVariables(b);
  }

  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  /**
   * Reduce ops do computation.
   * */
  bool isInitializingOp() const final { return false; }

  /**
   * Reduce ops are differentiable.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }

  void runSim(ISimState &ss) const final {
    runReplicatedSim(ss.simTensorMap());
  }

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  /**
   * The enum associated to this op's reduction type.
   * */
  virtual CommutativeOp cop() const = 0;

  std::string typeString() const final;

private:
  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}
  void computeDerivedVerifyValid() const final;
  void compute(const HostTensors &ins, const HostTensors &outs) const final;

  bool isValueDependent(InIndex, OutIndex) const final { return true; }

  // The dimensions to reduce:
  Dimensions dims_;
};

/**
 * Sum-reduction op.
 * */
class ReduceSum final : public Reduce {
public:
  ReduceSum(const State &s, const Dimensions &dims) : Reduce(s, dims) {}

  /**
   * Reduce all dimensions.
   * */
  ReduceSum(const State &s) : ReduceSum(s, s.inShape(0).dimensions()) {}

  UpOp cloneWithState(const State &) const final;

  /**
   * No inputs or outputs are required for backpropagation, as this op is a
   * linear transformation of its input.
   * */
  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  /**
   * Expand the gradient of the output, up to the shape of the input.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;
  CommutativeOp cop() const final { return CommutativeOp::Sum; }
};

/**
 * Min-reduction op.
 * */
class ReduceMin final : public Reduce {
public:
  ReduceMin(const State &s, const Dimensions &dims) : Reduce(s, dims) {}
  ReduceMin(const Shape &out);
  ReduceMin(const State &s) : Reduce(s) {}
  UpOp cloneWithState(const State &) const final;
  OptionalTensors bprop(const GradOpIns &) const final;
  std::vector<InIndex> autodiffRequiredIns() const final { return {0}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {0}; }
  CommutativeOp cop() const final { return CommutativeOp::Min; }
};

/**
 * Max-reduction op.
 * */
class ReduceMax final : public Reduce {
public:
  ReduceMax(const State &s, const Dimensions &dims) : Reduce(s, dims) {}
  ReduceMax(const Shape &out);
  ReduceMax(const State &s) : Reduce(s) {}
  UpOp cloneWithState(const State &) const final;
  OptionalTensors bprop(const GradOpIns &) const final;
  std::vector<InIndex> autodiffRequiredIns() const final { return {0}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {0}; }
  CommutativeOp cop() const final { return CommutativeOp::Max; }
};

/**
 * Product-reduction op.
 * */
class ReduceProduct final : public Reduce {
public:
  ReduceProduct(const State &s, const Dimensions &dims) : Reduce(s, dims) {}
  ReduceProduct(const Shape &out);
  ReduceProduct(const State &s) : Reduce(s) {}
  UpOp cloneWithState(const State &) const final;
  OptionalTensors bprop(const GradOpIns &) const final;
  std::vector<InIndex> autodiffRequiredIns() const final { return {0}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  CommutativeOp cop() const final { return CommutativeOp::Product; }
};

/**
 * An operation for reducing a tensor which is replicated across ipus. The
 * output tensor has the same shape as the input tensor, as the reduction is
 * done only in the implicit replication dimension.
 *
 * The reduction is done across subsets of the replicas. Specifically, the
 * replicas are partitioned into equally sized groups (\sa StridedPartition)
 * and the reduction is done independently in these groups.
 * */
class ReduceAcrossReplicas : public WithoutCalleesTensorCentric {
public:
  ReduceAcrossReplicas(const Op::State &s,
                       const StridedPartition &stridePartition)
      : WithoutCalleesTensorCentric(s), grouping_(stridePartition) {}

  bool isValueDependent(InIndex, OutIndex) const final { return true; }

  /**
   * An object which defines the replica groupings to reduce across.
   * */
  const StridedPartition &grouping() const { return grouping_; }

private:
  /**
   * A unary elementwise op does computation, and is therefore not an
   * 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  /**
   * The output is not a reference to a tensor in another graph. (\sa the
   * RefFrom op class).
   * */
  void resetRootRef(OutIndex, const TensorId &) final { invalid(); }
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  // Invalid, as runSim is implemented directly.
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedVerifyValid() const final;
  void runSim(ISimState &htm) const final;

  virtual poprithms::compute::host::CommutativeOp cop() const = 0;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  // There are no input/output dependent attributes in this class or any
  // inheriting class, so there will not be any op-specific work related to
  // the removal of inputs/outputs:
  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  bool computeTypeSpecificEqualTo(const Op &rhs) const {

    // Note that a static_cast is safe here, as this private method is only
    // called once it has been verified that #rhs has the same type (typeid)
    // as this object.
    const auto &rhs_ = static_cast<const ReduceAcrossReplicas &>(rhs);
    return grouping_ == rhs_.grouping_;
  }

  StridedPartition grouping_;
};

class ReduceAcrossReplicasOutplace : public ReduceAcrossReplicas {
public:
  ReduceAcrossReplicasOutplace(const State &s,
                               const StridedPartition &stridePartition)
      : ReduceAcrossReplicas(s, stridePartition) {}

private:
  bool aliases(InIndex, OutIndex) const final { return false; }
  bool modifies(InIndex) const final { return false; }
  HostTensors initializeOut(const HostTensors &) const final;

  void growAliasMapper(MemoryAliasMapper &mam) const final {
    // create new variables for the outputs.
    createVariables(mam);
  }
};

class ReduceAcrossReplicasInplace_ : public ReduceAcrossReplicas {
public:
  ReduceAcrossReplicasInplace_(const State &s,
                               const StridedPartition &stridePartition)
      : ReduceAcrossReplicas(s, stridePartition) {}

private:
  bool aliases(InIndex, OutIndex) const final { return true; }
  bool modifies(InIndex) const final { return true; }
  HostTensors initializeOut(const HostTensors &) const final;

  /**
   * Create a new variable/allocation in the alias::Graph corresponding to the
   * output of this op.
   * */
  void growAliasMapper(MemoryAliasMapper &mam) const final {
    createAlias(mam, inTensorId(0));
  }
};

/**
 * The gradient of a replica sum reduction with groupings #g is a replica sum
 * reduction with groupings #g, applied to the gradients of the outputs of the
 * forward reduction.
 * */
class SumAcrossReplicaAutodiffer {
public:
  // No input or output values from the forward reduction are required (just
  // like an 'add' operation).
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }
  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor> backpropagate(
      const poprithms::autodiff::automatic::OpIn<Tensor, OptionalTensor> &gIn,
      const Args &...) {
    return {gIn.gradOfOutput(0).reduceSumAcrossReplicas()};
  }
};

/**
 * Inplace sum-reduction across replicas.
 * */
class ReduceSumAcrossReplicas_ final
    : public WithAutodiff<SumAcrossReplicaAutodiffer,
                          ReduceAcrossReplicasInplace_> {
public:
  std::string typeString() const final { return "ReduceSumAcrossReplicas_"; }
  ReduceSumAcrossReplicas_(const State &s,
                           const StridedPartition &stridePartition)
      : WithAutodiff(s, stridePartition) {}

  poprithms::compute::host::CommutativeOp cop() const final {
    return poprithms::compute::host::CommutativeOp::Sum;
  }

  UpOp cloneWithState(const Op::State &s) const final {
    return std::unique_ptr<ReduceSumAcrossReplicas_>(
        new ReduceSumAcrossReplicas_(s, grouping()));
  }
};

/**
 * Non-inplace sum-reduction across replicas.
 * */
class ReduceSumAcrossReplicas final
    : public WithAutodiff<SumAcrossReplicaAutodiffer,
                          ReduceAcrossReplicasOutplace> {

public:
  ReduceSumAcrossReplicas(const Op::State &s,
                          const StridedPartition &stridePartition)
      : WithAutodiff(s, stridePartition) {}
  std::string typeString() const final { return "ReduceSumAcrossReplicas"; }

  UpOp cloneWithState(const Op::State &s) const final {
    return std::unique_ptr<ReduceSumAcrossReplicas>(
        new ReduceSumAcrossReplicas(s, grouping()));
  }

  poprithms::compute::host::CommutativeOp cop() const final {
    return poprithms::compute::host::CommutativeOp::Sum;
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
