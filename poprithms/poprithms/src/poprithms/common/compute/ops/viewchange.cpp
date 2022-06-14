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
#include <poprithms/util/stringutil.hpp>

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
 * DimShuffle_
 * */
HostTensors DimShuffle_::initializeOut(const HostTensors &ins) const {
  return {ins[0].dimShuffle_(permutation())};
}

void DimShuffle_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  if (p_.size() != inRank(0)) {
    std::ostringstream oss;
    oss << "The permutation of the DimShuffle_ has size " << p_.size()
        << " but the inpute tensor has rank " << inRank(0) << '.';
    throw error(oss.str());
  }
}

void DimShuffle_::growAliasMapper(MemoryAliasMapper &b) const {
  b.insert({b.graph().dimShuffle(b.id(inTensorId(0)), permutation())},
           outTensorIds());
}

bool DimShuffle_::computeTypeSpecificEqualTo(const Op &rhs) const {
  return p_ == static_cast<const DimShuffle_ &>(rhs).p_;
}

bool DimShuffle_::isIdentity(const Shape &in0,
                             const Shape &out0,
                             const Permutation &p) {

  /**
   * Does the permutation #p have no effect when applied to a tensor of shape
   * #in0?
   * */
  return (in0 == out0) && in0.dimShufflePreservesOrder(p);
}

UpOp DimShuffle_::cloneWithState(const State &s) const {
  return std::make_unique<DimShuffle_>(s, permutation());
}

std::string DimShuffle_::typeString() const {
  return poprithms::util::cat::strcat("DimShuffle_(p=", permutation(), ')');
}

DisjointRegions DimShuffle_::applyTo(const Region &r) const {
  return r.dimShuffle(permutation());
}

OptionalTensorIds DimShuffle_::backpropagate(Graph &g,
                                             const GradOpInIds &gIns) const {

  // Move from the tensor id based paradigm to a tensor based paradigm:
  GradOpIns gIn(g, gIns);

  auto outTensor = gIn.gradOfOutput(0).dimShuffle_(permutation().inverse());

  return {outTensor.id()};
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

void Reshape_::growAliasMapper(MemoryAliasMapper &b) const {
  b.insert({b.graph().reshape(b.id(inTensorId(0)), outShape(0))},
           outTensorIds());
}

bool Reshape_::isIdentity(const Shape &in0, const Shape &out0) {
  return in0 == out0;
}

/**
 * Reverse_
 * */
HostTensors Reverse_::initializeOut(const HostTensors &ins) const {
  return {ins[0].reverse_(dimensions().get())};
}

bool Reverse_::isIdentity(const Shape &in0,
                          const Shape &,
                          const Dimensions &dims) {

  // Does reversing a tensor a shape #in0 along dimensions #dims have no
  // effect?
  return in0.reversePreservesOrder(dims);
}

UpOp Reverse_::cloneWithState(const State &s) const {
  return std::make_unique<Reverse_>(s, dimensions());
}

Reverse_::Reverse_(const Op::State &s, const Dimensions &dimensions)
    : UnaryViewChange_(s), dimensions_(dimensions) {}

std::string Reverse_::typeString() const {
  return poprithms::util::cat::strcat("Reverse_(dims=", dimensions(), ')');
}

void Reverse_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});
}

DisjointRegions Reverse_::applyTo(const Region &r) const {
  return r.reverse(dimensions().get());
}

bool Reverse_::computeTypeSpecificEqualTo(const Op &rhs) const {
  return dimensions_ == static_cast<const Reverse_ &>(rhs).dimensions_;
}

void Reverse_::growAliasMapper(MemoryAliasMapper &b) const {
  b.insert({b.graph().reverse(b.id(inTensorId(0)), dimensions().get())},
           outTensorIds());
}

OptionalTensorIds Reverse_::backpropagate(Graph &g,
                                          const GradOpInIds &gIns) const {

  GradOpIns gIn(g, gIns);
  auto t0 = gIn.gradOfOutput(0).reverse_(dimensions());
  OptionalTensor ot(t0);

  return {ot};
}

} // namespace compute
} // namespace common
} // namespace poprithms
