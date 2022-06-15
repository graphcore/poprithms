// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/ops/reduce.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Reduce
 * */
HostTensors Reduce::initializeOut(const HostTensors &) const {
  return badValOuts();
}

void Reduce::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  if (inShape(0).rank_u64() != outShape(0).rank_u64()) {
    std::ostringstream oss;
    oss << "The rank of the input is " << inShape(0).rank_u64()
        << ", and the rank of the output is " << outShape(0).rank_u64()
        << ", they should be the same for a reduce op. ";
    throw error(oss.str());
  }
}

bool Reduce::computeTypeSpecificEqualTo(const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const Reduce &>(rhs);
  return dimensions() == rhs_.dimensions();
}

std::string Reduce::typeString() const {
  std::ostringstream oss;
  oss << "Reduce" << cop() << "(dimensions=" << dimensions() << ")";
  return oss.str();
}
void Reduce::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].reduce(outShape(0), cop()));
}

/**
 * ReduceSum
 * */

UpOp ReduceSum::cloneWithState(const State &s) const {
  return std::make_unique<ReduceSum>(s, dimensions());
}

OptionalTensors ReduceSum::bprop(const GradOpIns &gIn) const {

  (void)gIn;
  unimplemented("ReduceSum::bprop");

  // TODO(T64299). should this expand be inplace? Should reconsider all of the
  // aliasing backprop tensors.
  // auto t0 = gIn.gradOfOutput(0).expand_(inShape(0));
  // OptionalTensor t1(t0);
  // return {t1};
}

/**
 * ReduceMin
 * */

UpOp ReduceMin::cloneWithState(const State &s) const {
  return std::make_unique<ReduceMin>(s, dimensions());
}

OptionalTensors ReduceMin::bprop(const GradOpIns &) const {
  unimplemented("ReduceMin::bprop");
}

/**
 * ReduceMax
 * */

UpOp ReduceMax::cloneWithState(const State &s) const {
  return std::make_unique<ReduceMax>(s, dimensions());
}

OptionalTensors ReduceMax::bprop(const GradOpIns &) const {
  unimplemented("ReduceMax::bprop");
}

/**
 * ReduceProduct
 * */

UpOp ReduceProduct::cloneWithState(const State &s) const {
  return std::make_unique<ReduceProduct>(s, dimensions());
}

OptionalTensors ReduceProduct::bprop(const GradOpIns &) const {
  unimplemented("ReduceProduct::bprop");
}

} // namespace compute
} // namespace common
} // namespace poprithms
