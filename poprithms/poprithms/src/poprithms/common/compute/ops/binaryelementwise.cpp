// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/binaryelementwise.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

void BooleanBinaryElementwiseOutplace::boolReturnAutodiff() const {
  std::ostringstream oss;
  oss << "The op " << *this
      << " returns a boolean tensor, and so is not differentiable. ";
  throw error(oss.str());
}

/**
 * Non-aliasing binary elementwise op.
 *
 * Design note: why do we have inplace and output versions? The view-changing
 * ops are only inplace, to do an 'outplace' view-change a copy must be
 * inserted manually. Why is this pattern not followed for elementwise ops
 * too, it does mean less ops and code? The main reason for this is that (1)
 * backends might provide different APIs for inplace and outplace and (2) SSA
 * compute ops might be useful. This might change though, we might remove
 * outplace elementwise ops.
 * */

HostTensors
BinaryElementwiseOutplace::initializeOut(const HostTensors &) const {
  return badValOuts();
}

void BinaryElementwiseOutplace::simpleBinaryElementwiseOutplaceVerifyValid()
    const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      2, 1, {OpVerifier::Att::SameDType, OpVerifier::Att::SameDeviceType});

  if (inShape(0).numpyBinary(inShape(1)) != outShape(0)) {
    throw error("Input shapes do not combine with numpy broadcasting to "
                "output shape.");
  }
}

void BinaryElementwiseOutplace::growAliasMapper(
    MemoryAliasMapper &mam) const {
  createVariables(mam);
}

void BinaryElementwiseInplace_::growAliasMapper(
    MemoryAliasMapper &mam) const {
  createAlias(mam, inTensorId(0));
}

HostTensors
BinaryElementwiseInplace_::initializeOut(const HostTensors &ins) const {
  return {ins[0]};
}

void BinaryElementwiseInplace_::simpleBinaryElementwiseInplaceVerifyValid()
    const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      2, 1, {OpVerifier::Att::SameDType, OpVerifier::Att::SameDeviceType});
  inShape(0).assertNumpyDominates(inShape(1));
}

void Add::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0] + ins[1]);
}

UpOp Add::cloneWithState(const State &s) const {
  return std::make_unique<Add>(s);
}

UpOp Add_::cloneWithState(const State &s) const {
  return std::make_unique<Add_>(s);
}

void Mul::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].mul(ins[1]));
}

UpOp Mul::cloneWithState(const State &s) const {
  return std::make_unique<Mul>(s);
}

void BinaryElementwiseInplace_::noInplaceAutodiff() const {
  std::ostringstream oss;
  oss << "Attempt to backpropagate through inplace op " << *this
      << " is invalid. This is because an input value of " << *this
      << " is not available, having been modified inplace. "
      << "Consider for example c = a.mul_(b): "
      << "To compute db requires dc and a. "
      << "But the value of a gets updated during the inplace multiplication.";
  throw error(oss.str());
}

void Mul_::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].mul_(ins[1]);
}

UpOp Mul_::cloneWithState(const State &s) const {
  return std::make_unique<Mul_>(s);
}

bool Pow_::gradientPropagates(OutIndex, InIndex) const {
  noInplaceAutodiff();
}
poprithms::common::compute::OptionalTensors
Pow_::bprop(const GradOpIns &) const {
  noInplaceAutodiff();
}

std::vector<InIndex> Pow_::autodiffRequiredIns() const {
  noInplaceAutodiff();
}
std::vector<OutIndex> Pow_::autodiffRequiredOuts() const {
  noInplaceAutodiff();
}

/**
 * Div
 * */
void Div::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].divide(ins[1]));
}

UpOp Div::cloneWithState(const State &s) const {
  return std::make_unique<Div>(s);
}

/**
 * Div_
 * */
void Div_::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].divide_(ins[1]);
}

UpOp Div_::cloneWithState(const State &s) const {
  return std::make_unique<Div_>(s);
}

/**
 * Pow
 * */
void Pow::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].pow(ins[1]));
}
UpOp Pow::cloneWithState(const State &s) const {
  return std::make_unique<Pow>(s);
}

/**
 * Pow_
 * */
void Pow_::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].pow_(ins[1]);
}
UpOp Pow_::cloneWithState(const State &s) const {
  return std::make_unique<Pow_>(s);
}

/**
 * GreaterThan
 * */
void GreaterThan::compute(const HostTensors &ins,
                          const HostTensors &outs) const {
  outs[0].update_((ins[0] > ins[1]));
}

void EqualTo::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_((ins[0] == ins[1]));
}

/**
 * Sub
 * */
void Sub::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0] - ins[1]);
}

UpOp Sub::cloneWithState(const State &s) const {
  return std::make_unique<Sub>(s);
}

/**
 * Sub_
 * */
void Sub_::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].subtract_(ins[1]);
}

UpOp Sub_::cloneWithState(const State &s) const {
  return std::make_unique<Sub_>(s);
}

void BooleanBinaryElementwiseOutplace::
    simpleBooleanBinaryElementwiseInplaceVerifyValid() const {

  OpVerifier(*this).verifyNonVariadicFromAtts(
      2, 1, {OpVerifier::Att::InsSameDType, OpVerifier::Att::SameDeviceType});

  if (outDType(OutIndex(0)) != DType::Boolean) {
    std::ostringstream oss;
    oss << "invalid GreaterThan output in GreaterThan constructor. "
        << "State's output type is " << outDType(0);
    oss << ". Expected output type to be Boolean";
    throw error(oss.str());
  }

  if (inShape(0).numpyBinary(inShape(1)) != outShape(0)) {
    throw error("Input shapes do not combine with numpy broadcasting to "
                "output shape.");
  }
}

/**
 * CopyFrom_
 * */
UpOp CopyFrom_::cloneWithState(const State &s) const {
  return std::make_unique<CopyFrom_>(s);
}

void CopyFrom_::compute(const HostTensors &ins,
                        const HostTensors &outs) const {
  outs[0].update_(ins[Source().get()]);
}

} // namespace compute
} // namespace common
} // namespace poprithms
