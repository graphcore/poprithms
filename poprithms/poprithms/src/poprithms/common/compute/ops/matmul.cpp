// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/ops/matmul.hpp>

namespace poprithms {
namespace common {
namespace compute {

bool MatMul::computeTypeSpecificEqualTo(const Op &rhs) const {
  const auto &mmRhs = static_cast<const MatMul &>(rhs);
  return options() == mmRhs.options();
}

void MatMul::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].matmul(ins[1]).to(outDType(0)));
}

HostTensors MatMul::initializeOut(const HostTensors &) const {
  return badValOuts();
}

MatMul::MatMul(const State &s, const MatMulOptions &mmos)
    : WithAutodiff(s), matMulOptions_(mmos) {}

void MatMul::computeDerivedVerifyValid() const {

  OpVerifier(*this).verifyNonVariadicFromAtts(
      2, // number of inputs
      1, // number of outputs
      {OpVerifier::Att::InsSameDType,
       OpVerifier::Att::SameDevice,
       // currently only floating point can be used in matrix multiplication
       // on ipu:
       OpVerifier::Att::FloatIfIpu});

  // inputs must be rank-3 tensors.
  verifyRank(InIndex(0), 3);
  verifyRank(InIndex(1), 3);
}

std::unique_ptr<Op> MatMul::cloneWithState(const State &s) const {
  return std::make_unique<MatMul>(s, options());
}

} // namespace compute
} // namespace common
} // namespace poprithms
