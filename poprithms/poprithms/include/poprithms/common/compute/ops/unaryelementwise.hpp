// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_UNARYELEMENTWISE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_UNARYELEMENTWISE_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withautodiff.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::multiout::InIndices;
using common::multiout::OutIndices;

/**
 * An elementwise op with 1 input and 1 output.
 * */
class UnaryElementwise : public WithoutCalleesTensorCentric {
public:
  UnaryElementwise(const State &s) : WithoutCalleesTensorCentric(s) {}

  /**
   * A unary elementwise op does computation, and is therefore not an
   * 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  void runSim(SimTensorMap &htm) const final { runReplicatedSim(htm); }
  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  /**
   * The output is not a reference to a tensor in another graph. (\sa the
   * RefFrom op class).
   * */
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }
};

/**
 * An inplace unary elementwise op. The output is an alias of the input.
 * */
class UnaryElementwiseInplace_ : public UnaryElementwise {
public:
  UnaryElementwiseInplace_(const State &s) : UnaryElementwise(s) {}
  bool aliases(InIndex, OutIndex) const final { return true; }
  bool modifies(InIndex) const final { return true; }

  /**
   * Create a new variable/allocation in the alias::Graph corresponding to the
   * output of this op.
   * */
  void growAliasMapper(MemoryAliasMapper &b) const final {
    createAlias(b, inTensorId(0));
  }

  HostTensors initializeOut(const HostTensors &) const final;

  void computeDerivedVerifyValid() const final {
    OpVerifier(*this).verifyNonVariadicFromAtts(
        1, 1, {OpVerifier::Att::SameDType, OpVerifier::Att::SameDevice});
  }

  /**
   * The current implementation assumes that unary elementwise ops cannot do
   * backpropagation. This might need to change in the future.
   * */
  OptionalTensors bprop(const GradOpIns &) const final;
  bool gradientPropagates(OutIndex, InIndex) const final;
  std::vector<InIndex> autodiffRequiredIns() const final;
  std::vector<OutIndex> autodiffRequiredOuts() const final;

private:
  [[noreturn]] void noInplaceAutodiff() const;
};

class UnaryElementwiseOutplace : public UnaryElementwise {
public:
  void growAliasMapper(MemoryAliasMapper &b) const final {
    createVariables(b);
  }
  UnaryElementwiseOutplace(const State &s) : UnaryElementwise(s) {}
  bool aliases(InIndex, OutIndex) const final { return false; }
  bool modifies(InIndex) const final { return false; }
  HostTensors initializeOut(const HostTensors &) const final;

protected:
  // Not all elementwise ops are type preserving (cast, for example, has
  // different input and output types). This method can be used by those ops
  // which are type preserving.
  void typePreservingAssertValid() const {
    OpVerifier(*this).verifyNonVariadicFromAtts(
        1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});
  }
};

class Log final : public WithAutodiff<autodiff::automatic::LogAutodiffer,
                                      UnaryElementwiseOutplace> {

public:
  Log(const State &s) : WithAutodiff(s) {}
  std::string typeString() const final { return "Log"; }
  UpOp cloneWithState(const State &) const final;
  void compute(const HostTensors &, const HostTensors &) const final;
  bool computeTypeSpecificEqualTo(const compute::Op &) const final {
    return true;
  }

  void computeDerivedVerifyValid() const final {
    typePreservingAssertValid();
  }
};

class Log_ final : public UnaryElementwiseInplace_ {
public:
  Log_(const State &s) : UnaryElementwiseInplace_(s) {}
  UpOp cloneWithState(const State &) const final;
  std::string typeString() const final { return "Log_"; }
  void compute(const HostTensors &, const HostTensors &) const final;
  bool computeTypeSpecificEqualTo(const compute::Op &) const final {
    return true;
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
