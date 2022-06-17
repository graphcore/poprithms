// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/add.hpp>
#include <poprithms/common/compute/ops/elementwise.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {
/**
 * BinaryElementwiseOutplace
 * */
HostTensors
BinaryElementwiseOutplace::initializeOut(const HostTensors &) const {
  return badValOuts();
}

void BinaryElementwise::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      2, 1, {OpVerifier::Att::SameDType, OpVerifier::Att::SameDeviceType});
  binaryElementwiseDerivedVerifyValid();
}

void BinaryElementwiseOutplace::growAliasMapper(
    MemoryAliasMapper &mam) const {
  createVariables(mam);
}

/**
 * BinaryElementwiseInplace_
 * */

void BinaryElementwiseInplace_::growAliasMapper(
    MemoryAliasMapper &mam) const {
  createAlias(mam, inTensorId(0));
}

HostTensors
BinaryElementwiseInplace_::initializeOut(const HostTensors &ins) const {
  return {ins[0]};
}

std::string BinaryElementwiseInplace_::noInplaceAutodiff() const {
  std::ostringstream oss;
  oss << "Binary Inplace Op " << *this
      << "does not (currently) support autodiff. ";
  throw error(oss.str());
}

void BinaryElementwiseInplace_::binaryElementwiseDerivedVerifyValid() const {
  inShape(0).assertNumpyDominates(inShape(1));
  binaryElementwiseInplaceDerivedVerifyValid();
}

/**
 * Add
 * */
void Add::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0] + ins[1]);
}
UpOp Add::cloneWithState(const State &s) const {
  return std::make_unique<Add>(s);
}

OptionalTensors Add::addBackpropagate(const GradOpIns &gIn,
                                      const Shape &in0,
                                      const Shape &in1) {
  (void)gIn;
  (void)in0;
  (void)in1;
  throw error("unimplemented: Add::addBackpropagate");
  // TODO(T64299)
  //  return {gIn.gradOfOutput(0).reduce(in0, CommutativeOp::Sum),
  //          gIn.gradOfOutput(0).reduce(in1, CommutativeOp::Sum)};
}

OptionalTensors Add::bprop(const GradOpIns &gIn) const {
  return addBackpropagate(gIn, inShape(0), inShape(1));
}

/**
 * Add_
 * */

UpOp Add_::cloneWithState(const State &s) const {
  return std::make_unique<Add_>(s);
}

OptionalTensors Add_::bprop(const GradOpIns &gIn) const {
  return Add::addBackpropagate(gIn, inShape(0), inShape(1));
}

} // namespace compute
} // namespace common
} // namespace poprithms
