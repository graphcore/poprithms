// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_REFFROM_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_REFFROM_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>

namespace poprithms {
namespace common {
namespace compute {

class RefFrom final : public poprithms::common::compute::WithoutCallees {

public:
  RefFrom(const State &s, const TensorId &root);

  /**
   * Reset the root reference to #root.
   * */
  void resetRootRef(OutIndex o, const TensorId &root) final;

  /**
   * The tensor in a different sub-graph which the output of this op is a
   * reference (alias) of.
   * */
  TensorId rootRef(OutIndex) const final { return root_; }

  /**
   * Insert a new tensor into the alias graph of #mam which is an alias of the
   * root reference. Thus, the alias graph has an edge between tensors which
   * are different sub-graphs of the common::compute::Graph.
   * */
  void growAliasMapper(MemoryAliasMapper &b) const final;

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void computeDerivedVerifyValid() const final;

  /**
   * Any input index provided in the following methods is invalid, as this op
   * has no inputs.
   * */
  bool aliases(InIndex, OutIndex) const final { invalid(); }
  bool modifies(InIndex) const final { invalid(); }
  bool gradientPropagates(OutIndex, InIndex) const final { invalid(); }

  Shape shape() const { return outShape(0); }

  std::string typeString() const final;

  /**
   * RefFrom does no computation.
   * */
  bool isInitializingOp() const final { return true; }
  CodeLocation codeLocation() const final { return CodeLocation::None; }
  void runSim(ISimState &) const final {}
  void compute(const HostTensors &, const HostTensors &) const final {}

  std::unique_ptr<poprithms::common::compute::Op>
  cloneWithState(const State &) const final;

  bool computeTypeSpecificEqualTo(const compute::Op &rhs) const final;

  /**
   * Initialize the output to be an alias of the root reference.
   * */
  void initializeSimOut(SimTensorMap &) const final;

  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  OptionalTensorIds backpropagate(Graph &, const GradOpInIds &) const final {
    return {};
  }

private:
  // The root reference tensor, an alias of the output of this op, which is in
  // a different sub-graph.
  TensorId root_;

  HostTensors initializeOut(const HostTensors &) const final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
