// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

void UnaryElementwiseInplace_::noInplaceAutodiff() const {
  std::ostringstream oss;
  oss << "Unary Elementwise Inplace Ops do not currently support "
         "autodiff. "
      << "Failure for Op " << *this << '.';
  throw error(oss.str());
}

/**
 * UnaryElementwiseInplace_
 * */
HostTensors
UnaryElementwiseInplace_::initializeOut(const HostTensors &ins) const {
  return ins;
}

bool UnaryElementwiseInplace_::gradientPropagates(OutIndex, InIndex) const {
  noInplaceAutodiff();
}

OptionalTensors UnaryElementwiseInplace_::bprop(const GradOpIns &) const {
  noInplaceAutodiff();
}

std::vector<InIndex> UnaryElementwiseInplace_::autodiffRequiredIns() const {
  noInplaceAutodiff();
}
std::vector<OutIndex> UnaryElementwiseInplace_::autodiffRequiredOuts() const {
  noInplaceAutodiff();
}

/**
 * UnaryElementwiseOutplace
 * */
HostTensors
UnaryElementwiseOutplace::initializeOut(const HostTensors &) const {
  return {HostTensor::zeros(outDType(0), outShape(0))};
}

/**
 * Log
 * */
void Log::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].log());
}

UpOp Log::cloneWithState(const State &s) const {
  return std::make_unique<Log>(s);
}

/**
 * Log_
 * */
void Log_::compute(const HostTensors &, const HostTensors &outs) const {
  outs[0].log_();
}

UpOp Log_::cloneWithState(const State &s) const {
  return std::make_unique<Log_>(s);
}

} // namespace compute
} // namespace common
} // namespace poprithms
