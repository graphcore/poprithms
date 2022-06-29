// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_MATMUL_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_MATMUL_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withautodiff.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Matrix multiplication.
 * */
class MatMul final
    : public WithAutodiff<poprithms::autodiff::automatic::MatMulAutodiffer,
                          WithoutCalleesTensorCentric> {

public:
  /**
   * Inputs must be rank-3.
   * */
  MatMul(const State &, const MatMulOptions &);

  /**
   * Matrix multiplication specific options.
   * */
  const MatMulOptions &options() const { return matMulOptions_; }

private:
  MatMulOptions matMulOptions_;

  // Confirm that inputs are rank-3, and that the inputs have the same type.
  void computeDerivedVerifyValid() const final;

  void growAliasMapper(MemoryAliasMapper &b) const final {
    createVariables(b);
  }

  std::string typeString() const final { return "MatMul"; }

  /**
   * Matmul does computation, it is therefore not an 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  bool aliases(InIndex, OutIndex) const final { return false; }

  bool modifies(InIndex) const final { return false; }

  bool computeTypeSpecificEqualTo(const Op &) const final;

  std::unique_ptr<Op> cloneWithState(const State &s) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  /**
   * There is no aliasing between inputs and outputs, so this method
   * returns a single new allocation for the output tensor.
   * */
  HostTensors initializeOut(const HostTensors &) const final;

  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  void runSim(SimTensorMap &htm) const final { runReplicatedSim(htm); }

  /**
   * Host tensor matrix multiplication.
   * */
  void compute(const HostTensors &, const HostTensors &) const final;

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) {}
  void removeMachineDerivedOutputs(const ContiguousOutIndexSubset &) {}

  /**
   * Methods specific to an whose outputs are aliases of tensors in different
   * sub-graphs have. \sa RefFrom_.
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
