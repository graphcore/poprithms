// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_REDUCE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_REDUCE_HPP

#include <poprithms/common/compute/gradopins.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::compute::host::CommutativeOp;

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

  void runSim(SimTensorMap &htm) const final { runReplicatedSim(htm); }

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
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
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
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
