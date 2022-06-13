// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_VIEWCHANGE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_VIEWCHANGE_HPP

#include <poprithms/common/compute/ops/withoutcallees.hpp>

namespace poprithms {
namespace common {
namespace compute {

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

class Reshape_ final : public UnaryViewChange_ {
public:
  Reshape_(const Op::State &s) : UnaryViewChange_(s) {}

  /**
   * As this op class has no additional attributes, any reshape op which is
   * equivalent to it at the base op level (compute::Op) will always be
   * equivalent to it overall.
   * */
  bool computeTypeSpecificEqualTo(const compute::Op &) const final {
    return true;
  }

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

  void growAliasMapper(MemoryAliasMapper &b) const final {
    b.insert({b.graph().reshape(b.id(inTensorId(0)), outShape(0))},
             outTensorIds());
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
