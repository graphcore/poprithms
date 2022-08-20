// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/ops/dynamic.hpp>

namespace poprithms {
namespace common {
namespace compute {

void DynamicMulti::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

void DynamicMultiUpdateMax_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(3, 1, {});

  auto &&sliceableShape_ = sliceableShape();
  auto &&sliceShape_     = sliceShape();
  auto &&offsetShape_    = offsetShape();

  auto base = [&]() {
    std::ostringstream oss;
    oss << "\n   - Sliceable shape is " << sliceableShape_
        << "\n   - Slice shape is " << sliceShape_
        << "\n   - Offset shape is " << offsetShape_ << ".\n";
    return oss.str();
  };

  if (sliceableShape_.rank_i64() != 2 || sliceShape_.rank_i64() != 2 ||
      offsetShape_.rank_i64() != 1) {
    throw error(base() + " Expected sliceable and slice to be rank-2, and "
                         "offset to be rank-1.");
  }

  if (sliceableShape_.dim(1) != sliceShape_.dim(1)) {
    throw error(
        base() +
        " Expected sliceable and slice to be same size in dimension #1.");
  }

  if (offsetShape_.dim(0) != sliceShape_.dim(0)) {
    throw error(base() + " Expected offset dimension #0 to be the same as "
                         "slice dimension #0.");
  }

  OpVerifier(*this).verifySameTensorInfo(aliasIndex(), OutIndex(0));
}

OptionalTensors DynamicMultiUpdateMax_::bprop(const GradOpIns &gIns) const {

  OptionalTensors gradIns(nInTensors());

  auto gOut    = gIns.gradOfOutput(0);
  auto tOut    = gIns.output(0);
  auto tOffset = gIns.input(Offset());
  auto tSlice  = gIns.input(Slice());

  const auto S = gOut.dim(1);
  const auto M = tSlice.dim(0);

  auto scattered =
      Tensor::concat_({gOut, tOut}, 1)
          .dynamicMultiSlice(tOffset.reshape_({M, 1}), Dimensions{0}, {1})
          .reshape_({M, 2 * S});

  auto scatteredGOut = scattered.slice_(Dimension(1), 0, S);
  auto scatteredMax  = scattered.slice_(Dimension(1), S, 2 * S);

  auto mask = scatteredMax.equalTo(tSlice).to(tSlice.dtype());

  auto gSlice = tSlice.variable().zero_() + mask * scatteredGOut;

  gradIns.at(Slice().get()) = gSlice;

  return gradIns;
}

void DynamicMultiUpdateMax_::compute(const HostTensors &ins,
                                     const HostTensors &outs) const {

  auto tOffset = ins[Offset().get()];
  auto tOut    = outs[0];
  auto tSlice  = ins[Slice().get()];

  for (uint64_t i = 0; i < tSlice.dim(0); ++i) {
    auto index  = tOffset.getUnsigned32(i);
    auto update = tOut.at(index).max(tSlice.at(i));
    tOut.at_(index).update_(update);
  }
}

void DynamicMultiUpdate_::compute(const HostTensors &ins,
                                  const HostTensors &outs) const {

  auto tOffset = ins[Offset().get()];
  auto tOut    = outs[0];
  auto tSlice  = ins[Slice().get()].copy();

  for (uint64_t i = 0; i < tSlice.dim(0); ++i) {
    auto starts = tOffset.at(i).getUnsigned64Vector();
    tOut.updatePart_(tSlice.at(i), dimensions(), starts);
  }
}

DynamicMultiUpdate_::DynamicMultiUpdate_(const State &s,
                                         const Dimensions &dims)
    : DynamicMultiWithDimensions_(s, dims) {}

UpOp DynamicMultiUpdate_::cloneWithState(const State &s) const {
  return std::make_unique<DynamicMultiUpdate_>(s, dimensions());
}

std::string DynamicMultiUpdate_::typeString() const {
  return poprithms::util::cat::strcat(
      "DynamicMultiUpdate_(dims=", dimensions(), ')');
}

OptionalTensors DynamicMultiUpdate_::bprop(const GradOpIns &gIns) const {

  OptionalTensors gradIns(nInTensors());

  // The gradient of the output. This has the shape of 'sliceable'.
  auto gOut = gIns.gradOfOutput(0);

  // The offset tensor.
  auto tOffset = gIns.input(Offset());

  // The slice tensor gets a gradient.
  // (1) create of variable of the correct shape, type, etc.
  // (2) multi slice it (inplace).
  gradIns.at(Slice().get()) =
      gOut.variable(sliceShape())
          .dynamicMultiSlice_(gOut, tOffset, dimensions());

  return gradIns;
}

void DynamicMultiSlice_::compute(const HostTensors &ins,
                                 const HostTensors &outs) const {

  auto sliceable = ins[Sliceable().get()];
  auto offset    = ins[Offset().get()];
  auto slice     = ins[Slice().get()];

  for (uint64_t i = 0; i < slice.dim(0); ++i) {
    const auto lStarts = offset.at(i);
    const auto starts_ = lStarts.getUnsigned64Vector();
    const auto ends_ = sizes().addToDims(lStarts.getInt64Vector()).get_u64();
    auto subSlice    = sliceable.slice(dimensions(), starts_, ends_);
    outs[0].at_(i).update_(subSlice);
  }
}

DynamicMultiSlice_::DynamicMultiSlice_(const State &s, const Dimensions &dims)
    : DynamicMultiWithDimensions_(s, dims) {}

std::string DynamicMultiSlice_::typeString() const {
  return poprithms::util::cat::strcat(
      "DynamicMultiSlice_(dims=", dimensions(), ')');
}

UpOp DynamicMultiSlice_::cloneWithState(const State &s) const {
  return std::make_unique<DynamicMultiSlice_>(s, dimensions());
}

OptionalTensors DynamicMultiSlice_::bprop(const GradOpIns &gIns) const {

  OptionalTensors gradIns(nInTensors());

  // The gradient of the output. This has the shape of 'slice'.
  auto gOut = gIns.gradOfOutput(0);

  // The offset tensor.
  auto tOffset = gIns.input(Offset());

  // The sliceable tensor gets a gradient.
  // (1) create of variable of the correct shape, type, etc, and zero it.
  // (2) to a multi update on (1) from the gradient of the output.
  gradIns.at(Sliceable().get()) =
      gOut.variable(sliceableShape())
          .zero_()
          .dynamicMultiUpdate_(gOut, tOffset, dimensions());

  return gradIns;
}

bool DynamicMultiWithDimensions_::computeTypeSpecificEqualTo(
    const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const DynamicMultiWithDimensions_ &>(rhs);
  return rhs_.dims_ == dims_;
}

void DynamicMultiWithDimensions_::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(3, 1, {});

  auto &&offsetShape_    = offsetShape();
  auto &&sliceableShape_ = sliceableShape();
  auto &&sliceShape_     = sliceShape();

  if (sliceableShape_.rank_u64() + 1 != sliceShape_.rank_u64()) {
    std::ostringstream oss;
    oss << "The sliceable tensor has shape " << sliceableShape_
        << " and the slice tensor has shape " << sliceShape_
        << ". The slice tensor should have a rank of 1 more than"
        << " the sliceable tensor's.";
    throw error(oss.str());
  }

  if (offsetShape_.rank_u64() != 2) {
    std::ostringstream oss;
    oss << "The offset tensor should have rank 2. "
        << "It is " << offsetShape_ << '.';
    throw error(oss.str());
  }

  if (offsetShape_.dim(0) != sliceShape_.dim(0)) {
    std::ostringstream oss;
    oss << "Dimension #0 of the offset tensor and the slice tensor "
        << "should be the same. "
        << "But the offset tensor has shape " << offsetShape_
        << " and the slice tensor has shape " << sliceShape_ << '.';
    throw error(oss.str());
  }

  sliceableShape_.assertDynamicUpdate(
      sliceShape_.fromDim(1), dimensions(), offsetShape_.fromDim(1));

  if (offsetShape_.dim_u64(1) != dimensions().size()) {
    std::ostringstream oss;
    oss << "Dimension #1 of the offset tensor is " << offsetShape_.dim_u64(1)
        << ". This is different to the size of dimensions, "
        << dimensions().size() << ". The offset tensor has shape "
        << offsetShape_ << " and dimensions is " << dimensions() << ".";
    throw error(oss.str());
  }

  OpVerifier(*this).verifySameTensorInfo(aliasIndex(), OutIndex(0));
}

Shape DynamicMultiWithDimensions_::sizes() const {
  auto &&s0 = inShape(Slice());
  std::vector<int64_t> sizes_;
  sizes_.reserve(dims_.size());
  for (auto d : dimensions().get()) {
    sizes_.push_back(s0.dim(d + 1));
  }

  return Shape(std::move(sizes_));
}

std::vector<uint64_t> DynamicMultiWithDimensions_::sizes_u64() const {
  return sizes().get_u64();
}

Shape DynamicMultiWithDimensions_::getSlicedShape(const Shape &offsetShape,
                                                  const Shape &sliceableShape,
                                                  const Dimensions &dims,
                                                  const Shape &sizes) {

  auto get = [&]() {
    std::ostringstream oss;
    oss << "In call getSlicedShape"
        << "(offsetShape=" << offsetShape
        << ", sliceableShape=" << sliceableShape << ", dims=" << dims
        << ", sizes=" << sizes << "): ";
    return oss.str();
  };
  if (offsetShape.rank_u64() != 2) {
    throw error(get() + "rank of offsetShape must be 2.");
  }

  if (offsetShape.dim_u64(1) != dims.size()) {
    throw error(get() + "offsetShape.dim(1) != dims.size().");
  }

  if (sizes.rank_u64() != dims.size()) {
    throw error(get() + "sizes.rank() != dims.size().");
  }

  auto vOutShape = sliceableShape.prepend(offsetShape.dim(0)).get();
  for (uint64_t i = 0; i < dims.size(); ++i) {
    if (dims.at(i).get() >= sliceableShape.rank_u64()) {
      std::ostringstream oss;
      oss << get() << "dims.at(i) >= this->rank(). "
          << "Can only slice this tensor in dimensions [0, rank).";
      throw error(oss.str());
    }
    vOutShape[dims.at(i).get() + 1] = sizes.dim(i);
  }

  return vOutShape;
}

} // namespace compute
} // namespace common
} // namespace poprithms
