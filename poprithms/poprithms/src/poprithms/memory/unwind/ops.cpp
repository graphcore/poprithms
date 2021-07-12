// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <memory/unwind/error.hpp>
#include <memory/unwind/ops.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

namespace {

template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &ts) {
  poprithms::util::append(ost, ts);
  return ost;
}

// Some variadic string sugar
void append(std::ostream &) {}
template <class Arg0, class... Args>
void append(std::ostream &oss, Arg0 &&arg0, Args &&... args) {
  oss << arg0;
  return append(oss, args...);
}
template <class... Args> std::string strcat(Args &&... args) {
  std::ostringstream oss;
  append(oss, args...);
  return oss.str();
}

} // namespace

// -------- //
//  Concat  //
// -------- //

void Concat::extendBwd(Chain &c, InIndex i, OutIndex o) const {
  verify(i, o, "extendBwd");
  c.slice(getLowerSlice(i), getUpperSlice(i));
}

void Concat::extendFwd(Chain &c, InIndex i, OutIndex o) const {
  verify(i, o, "extendFwd");
  c.settFillInto(
      Region::fromBounds(outShape(0), getLowerSlice(i), getUpperSlice(i)));
}

std::vector<int64_t> Concat::getLowerSlice(InIndex i) const {
  std::vector<int64_t> x(outRank(0), 0LL);
  x[axis()] = partitionPoints_[i.get()];
  return x;
}
std::vector<int64_t> Concat::getUpperSlice(InIndex i) const {
  auto x    = outShape(0).get();
  x[axis()] = partitionPoints_[i.get() + 1];
  return x;
}
std::string Concat::typeString() const {
  return strcat("Concat(axis=", axis(), ')');
}
bool Concat::unwindTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Concat &>(rhs);
  return axis() == rhs_.axis();
}

bool MatMulSource::unwindTypeSpecificEqualTo(const Op &other) const {
  const auto &rhs__ = dynamic_cast<const MatMulSource &>(other);
  return lhs() == rhs__.lhs() && rhs() == rhs__.rhs();
}

// ------- //
//  Input  //
// ------- //
void Input::extendFwd(Chain &, InIndex, OutIndex) const {
  throw error("No Input::extendFwd implemented, as no valid InIndex");
}
void Input::extendBwd(Chain &, InIndex, OutIndex) const {
  throw error("No Input::extendBwd implemented, as no valid InIndex");
}

// ------------ //
//  SettSample  //
// ------------ //
std::string SettSample::typeString() const {
  std::ostringstream oss;
  oss << "SettSample(" << region().setts() << ')';
  return oss.str();
}
bool SettSample::unwindTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const SettSample &>(rhs);
  return region().equivalent(rhs_.region());
}

void SettSample::bwd(Chain &c) const { c.settFillInto(region()); }
void SettSample::fwd(Chain &c) const { c.settSample(region()); }

// ------------ //
//  DimShuffle  //
// ------------ //
void DimShuffle::bwd(Chain &c) const {
  c.dimShuffle(permutation().inverse());
}
void DimShuffle::fwd(Chain &c) const { c.dimShuffle(permutation()); }

std::string DimShuffle::typeString() const {
  return strcat("DimShuffle(permutation=", permutation(), ')');
}
bool DimShuffle::unwindTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const DimShuffle &>(rhs);
  return permutation() == rhs_.permutation();
}

// --------- //
//  Reverse  //
// --------- //
void Reverse::bwd(Chain &c) const { c.reverse(dimensions()); }
void Reverse::fwd(Chain &c) const { c.reverse(dimensions()); }

std::string Reverse::typeString() const {
  return strcat("Reverse(dimensions=", dimensions().get(), ")");
}
bool Reverse::unwindTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Reverse &>(rhs);
  return dimensions().get() == rhs_.dimensions().get();
}

// --------- //
//  Reshape  //
// --------- //
void Reshape::bwd(Chain &c) const { c.reshape(inShape(0)); }
void Reshape::fwd(Chain &c) const { c.reshape(outShape(0)); }

Reshape::Reshape(const State &st) : ViewChange1to1(st) {
  if (st.baseState.inShapes.size() != 1 ||
      st.baseState.outShapes.size() != 1) {
    throw error("Invalid reshape, expected 1 input and 1 output");
  }

  if (st.baseState.outShapes[0].nelms_u64() !=
      st.baseState.inShapes[0].nelms_u64()) {
    std::ostringstream oss;
    oss << "Invalid reshape, number of elements changes. "
        << "Cannot reshape from " << st.baseState.inShapes[0] << " to "
        << st.baseState.outShapes[0] << ". ";
    throw error(oss.str());
  }
}

// ------- //
// Barrier //
// ------- //
void BaseBarrier::extendFwd(Chain &, InIndex, OutIndex) const {
  throw error("No extendFwd for BaseBarrier");
}
void BaseBarrier::extendBwd(Chain &, InIndex, OutIndex) const {
  throw error("No extendBwd for BaseBarrier");
}

//  ------------  //
// ViewChange1to1 //
//  ------------  //
void ViewChange1to1::extendFwd(Chain &c, InIndex i, OutIndex o) const {
  Op::verify(i, o, "extendFwd");
  fwd(c);
}

void ViewChange1to1::extendBwd(Chain &c, InIndex i, OutIndex o) const {
  Op::verify(i, o, "extendBwd");
  bwd(c);
}

/////////////
// SumLike //
/////////////

SumLike::SumLike(const State &st, InIndex unwindIndex)
    : NonInput(st), unwindIndex_(unwindIndex) {
  if (st.baseState.inShapes.size() <= unwindIndex.get()) {
    std::ostringstream oss;
    oss << "Invalid number of inputs to SumLike constructor. "
        << "Number of inputs in State = " << st.baseState.inShapes.size()
        << ", while unwindIndex = " << unwindIndex << ". ";
    throw error(oss.str());
  }

  if (st.baseState.outShapes.size() != 1) {
    std::ostringstream oss;
    oss << "Expected exactly 1 output Shape in SumLike constructor, not "
        << st.baseState.outShapes.size() << '.';
    throw error(oss.str());
  }
  if (st.baseState.inShapes[unwindIndex.get()] != st.baseState.outShapes[0]) {
    std::ostringstream oss;
    oss << "Invalid Shape of input at unwindIndex (" << unwindIndex
        << ") of SumLike Op, " << st.baseState.inShapes[unwindIndex.get()]
        << ". It must be the same as the output Shape, "
        << st.baseState.outShapes[0] << ". "
        << "This design decision is taken for this unwinding project, "
        << "where we assume the output inherits its layout "
        << "from a single input. ";
    throw error(oss.str());
  }
}

void SumLike::extendFwd(Chain &, InIndex i, OutIndex o) const {
  Op::verify(i, o, "extendFwd");
  if (i != unwindIndex()) {
    throw error("Can only extendFwd at InIndex " + i +
                " for this SumLike Op");
  }
  // identity : no extension to Chain required.
}

bool SumLike::unwindTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const SumLike &>(rhs);
  return unwindIndex() == rhs_.unwindIndex();
}

std::string SumLike::typeString() const {
  return "SumLike(unwindIndex=" + unwindIndex() + ')';
}

void SumLike::extendBwd(Chain &, InIndex i, OutIndex o) const {
  Op::verify(i, o, "extendBwd");
  if (i != unwindIndex()) {
    throw error("Can only extendBwd at InIndex " + i +
                " for this SumLike Op");
  }
  // identity : no extension to Chain required.
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
