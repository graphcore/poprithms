// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

void UnaryElementwise::computeDerivedVerifyValid() const {

  switch (outType()) {
  case OutType::Preserving: {
    OpVerifier(*this).verifyNonVariadicFromAtts(
        1, 1, {OpVerifier::Att::SameDType, OpVerifier::Att::SameDevice});
    return;
  }

  case OutType::Bool: {
    OpVerifier(*this).verifyNonVariadicFromAtts(
        1, 1, {OpVerifier::Att::SameDevice});

    if (outDType(0) != ndarray::DType::Boolean) {
      throw error("Expected output of " + str() + " to be Boolean.");
    }
    return;
  }

  case OutType::Other: {
    OpVerifier(*this).verifyNonVariadicFromAtts(
        1, 1, {OpVerifier::Att::SameDevice});
    return;
  }

  default: {
    throw error("Unrecognised OutType.");
  }
  }

  if (inShape(0) != outShape(0)) {
    throw error("Expected input and output shapes of " + str() +
                " to be the same");
  }

  unaryElementwiseDerivedVerifyValid();
}

HostTensors
UnaryElementwiseInplace_::initializeOut(const HostTensors &ins) const {
  return ins;
}

HostTensors
UnaryElementwiseOutplace::initializeOut(const HostTensors &) const {
  return {HostTensor::zeros(outDType(0), outShape(0))};
}

std::string Log_::whyNoAutodiff() const {
  std::ostringstream oss;
  oss << "Inplace log op " << *this << " does not support autodiff, "
      << "as the input gets written to. "
      << "The grad could be computed from the output "
      << "(dIn = dOut*exp(-out)) "
      << "but numerical accuracy might be poor. ";
  throw error(oss.str());
}

void Log::unaryCompute(const HostTensor &i, const HostTensor &o) const {
  o.update_(i.log());
}

void Log_::unaryCompute(const HostTensor &, const HostTensor &o) const {
  o.log_();
}

void Cast::unaryCompute(const HostTensor &ins, const HostTensor &outs) const {
  outs.update_(ins.to(outDType(0)));
}

bool Cast::gradientPropagates(OutIndex, InIndex) const {
  if (poprithms::ndarray::isFixedPoint(outDType(0))) {
    std::ostringstream oss;
    oss << "Failure detected in gradientPropagates of " << *this;
    oss << ". Expected the output of the cast to "
        << "be a floating point tensor. ";
    throw error(oss.str());
  }
  // propagates if the input is floating point.
  return !poprithms::ndarray::isFixedPoint(inDType(0));
}

OptionalTensors Cast::bprop(const GradOpIns &gIn) const {
  return {gIn.gradOfOutput(0).to(inDType(0))};
}

std::string Fill_::typeString() const {
  return poprithms::util::cat::strcat("Fill_(", val_.valueAsStr(0), ")");
}

UpOp Fill_::cloneWithState(const State &s) const {
  return std::make_unique<Fill_>(s, val_);
}

bool Fill_::computeTypeSpecificEqualTo(const Op &rhs) const {
  const Fill_ &rhs_ = static_cast<const Fill_ &>(rhs);
  return val_.numericallyIdenticalTo(rhs_.val_);
}

void Fill_::unaryElementwiseDerivedVerifyValid() const {
  if (val_.rank_u64() != 0) {
    std::ostringstream oss;
    oss << "Expected the fill value of " << *this << ", which is " << val_
        << ", to have only 1 element and be of rank-0. ";
    throw error(oss.str());
  }
}

} // namespace compute
} // namespace common
} // namespace poprithms
