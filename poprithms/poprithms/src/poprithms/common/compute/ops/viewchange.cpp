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
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

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

void DimShuffle_::growAliasMapper(MemoryAliasMapper &mam) const {
  mam.insert({mam.graph().dimShuffle(mam.id(inTensorId(0)), permutation())},
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

OptionalTensors DimShuffle_::bprop(const GradOpIns &gIn) const {
  return {gIn.gradOfOutput(0).dimShuffle_(permutation().inverse())};
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

OptionalTensors Reshape_::bprop(const GradOpIns &gIn) const {
  return {gIn.gradOfOutput(0).reshape_(inShape(0))};
}

void Reshape_::growAliasMapper(MemoryAliasMapper &mam) const {
  mam.insert({mam.graph().reshape(mam.id(inTensorId(0)), outShape(0))},
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
    : ViewChange_(s), dimensions_(dimensions) {}

std::string Reverse_::typeString() const {
  return poprithms::util::cat::strcat("Reverse_(dims=", dimensions(), ')');
}

void Reverse_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});
}

bool Reverse_::computeTypeSpecificEqualTo(const Op &rhs) const {
  return dimensions_ == static_cast<const Reverse_ &>(rhs).dimensions_;
}

void Reverse_::growAliasMapper(MemoryAliasMapper &mam) const {
  mam.insert({mam.graph().reverse(mam.id(inTensorId(0)), dimensions().get())},
             outTensorIds());
}

OptionalTensors Reverse_::bprop(const GradOpIns &gIn) const {
  return {gIn.gradOfOutput(0).reverse_(dimensions())};
}

/**
 * Expand_
 * */
HostTensors Expand_::initializeOut(const HostTensors &ins) const {
  return {ins[0].expand_(outShape(0))};
}

UpOp Expand_::cloneWithState(const State &s) const {
  return std::make_unique<Expand_>(s);
}

Expand_::Expand_(const State &s) : ViewChange_(s) {
  inShape(0).assertCanExpandTo(outShape(0));
}

OptionalTensors Expand_::bprop(const GradOpIns &gIn) const {
  return {gIn.gradOfOutput(0).reduce(inShape(0), CommutativeOp::Sum)};
}

bool Expand_::computeTypeSpecificEqualTo(const Op &) const { return true; }

void Expand_::growAliasMapper(MemoryAliasMapper &mam) const {
  mam.insert({mam.graph().expand(mam.id(inTensorId(0)), outShape(0))},
             outTensorIds());
}

void Expand_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  // assert that the input shape can be expanded to the output shape.
  outShape(0).assertNumpyDominates(inShape(0));
}
/**
 * Slice_
 * */
HostTensors Slice_::initializeOut(const HostTensors &ins) const {
  return {ins[0].slice_(lower(), upper())};
}

Slice_::Slice_(const Op::State &s, const Lower &lower__, const Upper &upper__)
    : ViewChange_(s), lower_(lower__), upper_(upper__) {}

UpOp Slice_::cloneWithState(const State &s) const {
  return std::make_unique<Slice_>(s, lower(), upper());
}

std::string Slice_::typeString() const {
  std::ostringstream oss;
  oss << "Slice_(l=";
  poprithms::util::append(oss, lower());
  oss << ",u=";
  poprithms::util::append(oss, upper());
  oss << ')';
  return oss.str();
}
bool Slice_::computeTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Slice_ &>(rhs);
  return lower() == rhs_.lower() && upper() == rhs_.upper();
}

void Slice_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  inShape(0).assertSliceBoundsAreValid(lower(), upper());
}

using GradOpIns = GradOpIns;
OptionalTensors Slice_::bprop(const GradOpIns &gIn) const {

  // The gradient constant a broadcast constant 0. If this is a problem. we
  // can consider using a variable which is set to zero every iteration.

  return {gIn.gradOfOutput(0).padWithBroadcastConstZero_(lower(), upper())};
}

void Slice_::growAliasMapper(MemoryAliasMapper &mam) const {
  mam.insert({mam.tensor(inTensorId(0)).slice(lower(), upper()).id()},
             outTensorIds());
}

UpOp Concat_::cloneWithState(const State &s) const {
  return std::make_unique<Concat_>(s, axis());
}

void Concat_::growAliasMapper(MemoryAliasMapper &mam) const {

  auto aOut = mam.graph().concat(mam.ids(inTensorIds()), axis());
  mam.insert({mam.graph().tensor(aOut).id()}, outTensorIds());
}

HostTensors Concat_::initializeOut(const HostTensors &ins) const {
  return {HostTensor::concat_(ins, axis())};
}
std::string Concat_::typeString() const {
  return poprithms::util::cat::strcat("Concat_(axis=", axis(), ')');
}

void Concat_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyFromAtts(
      {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});
}

bool Concat_::computeTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Concat_ &>(rhs);

  // Note that we don't compare the attribute partitionPoints_, as this is
  // derived directly from axis. So if 2 concatenation ops have the same
  // axis, they have the same partitionPoints_.
  return axis() == rhs_.axis();
}

std::vector<int64_t> Concat_::lowerSlice(InIndex i) const {
  std::vector<int64_t> x(outShape(0).rank_u64(), 0LL);
  x[axis()] = partitionPoints_[i.get()];
  return x;
}

std::vector<int64_t> Concat_::upperSlice(InIndex i) const {
  auto x    = outShape(OutIndex(0)).get();
  x[axis()] = partitionPoints_[i.get() + 1];
  return x;
}

Tensors Concat_::slice_(const Tensor &toSlice) const {
  if (outShape(0) != toSlice.shape()) {
    std::ostringstream oss;
    oss << "Expected a tensor of shape " << outShape(0) << ", not "
        << toSlice.shape();
    throw error(oss.str());
  }
  Tensors inSlices;
  inSlices.reserve(nInTensors());
  for (InIndex i = 0; i < nInTensors(); ++i) {
    inSlices.push_back(toSlice.slice_(lowerSlice(i), upperSlice(i)));
  }
  return inSlices;
}

OptionalTensors Concat_::bprop(const GradOpIns &gIns) const {
  return OptionalTensor::fromTensors(slice_(gIns.gradOfOutput(0)));
}

void ViewChange_::resetRootRef(OutIndex, const TensorId &) { invalid(); }

} // namespace compute
} // namespace common
} // namespace poprithms
