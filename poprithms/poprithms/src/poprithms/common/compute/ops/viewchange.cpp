// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/common/compute/gradopins.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/opverifier.hpp>

namespace poprithms {
namespace common {
namespace compute {

DisjointRegions
UnaryViewChange_::apply(const std::vector<Region> &regions) const {

  if (regions.size() != 1) {
    std::ostringstream oss;
    oss << "The op " << *this
        << " is a UnaryViewChange_ op and so has only 1 input. "
        << "The number of regions in apply should therefore be 1, but it is "
        << regions.size() << '.';
    throw error(oss.str());
  }
  return applyTo(regions[0]);
}

void Reshape_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  if (outShape(0).nelms() != inShape(0).nelms()) {
    std::ostringstream oss;
    oss << "Invalid reshape, number of elements not preserved. "
        << "Input " << inShape(0) << " has " << inShape(0).nelms()
        << " elements and output " << outShape(0) << " has "
        << outShape(0).nelms() << " elements.";
    throw error(oss.str());
  }
}

/**
 * Reshape_
 * */
HostTensors Reshape_::initializeOut(const HostTensors &ins) const {
  // An "inplace" reshape of the host tensor. i.e. output is an alias of the
  // input.
  return {ins[0].reshape_(outShape(0))};
}

UpOp Reshape_::cloneWithState(const State &s) const {
  return std::make_unique<Reshape_>(s);
}

OptionalTensorIds Reshape_::backpropagate(Graph &g,
                                          const GradOpInIds &gIns) const {

  GradOpIns gIn(g, gIns);

  // Reshape (inplace) the gradient of the output to have the shape of the
  // input.
  return {gIn.gradOfOutput(0).reshape_(inShape(0)).id()};
}

DisjointRegions Reshape_::applyTo(const Region &r) const {
  return r.reshape(outShape(0));
}

} // namespace compute
} // namespace common
} // namespace poprithms
