// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <memory/inplace/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/color.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

namespace {

static_assert(std::is_nothrow_move_constructible<
                  poprithms::memory::inplace::DimShuffle>::value,
              "Expect DimShuffle to be nothrow move constructible");

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

using UpOp = std::unique_ptr<Op>;
template <typename OP> UpOp mu(const OP *const derived) {
  return std::make_unique<OP>(*derived);
}
} // namespace

// -------- //
//  Concat  //
// -------- //

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
bool Concat::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Concat &>(rhs);
  return axis() == rhs_.axis();
}
std::vector<alias::TensorId>
Concat::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  for (auto x : inTensorIds()) {
    m.toAliasGraphId(x);
  }

  return {g.concat(m.toAliasGraphIds(inTensorIds()), axis())};
}
UpBop Concat::clone() const { return mu<Concat>(this); }

// ------- //
//  Alloc  //
// ------- //
std::string Alloc::typeString() const {
  return strcat("Alloc(color=", color(), ')');
}

bool Alloc::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Alloc &>(rhs);
  return color() == rhs_.color();
}

UpBop Alloc::clone() const { return mu<Alloc>(this); }

std::vector<alias::TensorId>
Alloc::typeSpecificGrow(alias::Graph &g, const TensorMap &) const {
  AliasTensorIds ids;
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    ids.push_back(g.allocate(outShape(o), color()));
  }
  return ids;
}

// --------------- //
//  UnaryModifier  //
// --------------- //
std::vector<alias::TensorId>
UnaryModifier::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.identity(m.toAliasGraphId(inTensorId(0)))};
}
UpBop UnaryModifier::clone() const { return mu<UnaryModifier>(this); }

// ------------ //
//  SettSample  //
// ------------ //
std::string SettSample::typeString() const {
  std::ostringstream oss;
  oss << "SettSample(" << region().setts() << ')';
  return oss.str();
}
bool SettSample::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const SettSample &>(rhs);
  return region().equivalent(rhs_.region());
}
std::vector<alias::TensorId>
SettSample::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.settSample(m.toAliasGraphId(inTensorId(0)), region())};
}
UpBop SettSample::clone() const { return mu<SettSample>(this); }

// ------------ //
//  DimShuffle  //
// ------------ //
std::string DimShuffle::typeString() const {
  return strcat("DimShuffle(permutation=", permutation(), ')');
}
bool DimShuffle::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const DimShuffle &>(rhs);
  return permutation() == rhs_.permutation();
}
std::vector<alias::TensorId>
DimShuffle::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.dimShuffle(m.toAliasGraphId(inTensorId(0)), permutation())};
}
UpBop DimShuffle::clone() const { return mu<DimShuffle>(this); }

// --------- //
//  Reverse  //
// --------- //
std::string Reverse::typeString() const {
  return strcat("Reverse(dimensions=", dimensions().get(), ")");
}
bool Reverse::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Reverse &>(rhs);
  return dimensions().get() == rhs_.dimensions().get();
}
std::vector<alias::TensorId>
Reverse::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.reverse(m.toAliasGraphId(inTensorId(0)), dimensions().get())};
}
UpBop Reverse::clone() const { return mu<Reverse>(this); }

// --------- //
//  Reshape  //
// --------- //
std::vector<alias::TensorId>
Reshape::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.reshape(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
UpBop Reshape::clone() const { return mu<Reshape>(this); }

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

// --------- //
//  Expand   //
// --------- //
std::vector<alias::TensorId>
Expand::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.expand(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
UpBop Expand::clone() const { return mu<Expand>(this); }

// ----- //
// Multi //
// ----- //
Multi::Multi(const State &st, const CrossLinks &m) : Op(st), mapping_(m) {
  const auto nIn  = st.baseState.inIds.size();
  const auto nOut = st.baseState.outShapes.size();

  // 1) Verify that all indices are valud
  for (const auto &crossAlias : m) {
    if (crossAlias.in() >= nIn || crossAlias.out() >= nOut) {
      std::ostringstream oss;
      oss << "Invalid CrossLink " << crossAlias << ", where this Multi has "
          << nIn << " inputs and " << nOut << " outputs. ";
      throw error(oss.str());
    }
  }

  std::vector<bool> inAliasSeen(nIn, false);
  std::vector<bool> outAliasSeen(nOut, false);

  // 2) each input and each output can be in at most 1 aliasing relationship
  for (const auto &crossAlias : m) {
    if (crossAlias.isModifying() || crossAlias.isAliasing()) {
      if (inAliasSeen[crossAlias.in_u64()]) {
        std::ostringstream oss;
        oss << "Invalid crossAliases : " << m
            << ". Inputs can only appear in 1 aliasing CrossLink. ";
        throw error(oss.str());
      }
      if (outAliasSeen[crossAlias.out_u64()]) {
        std::ostringstream oss;
        oss << "Invalid crossAliases : " << m
            << ". Outputs can only appear in 1 aliasing CrossLink. ";
        throw error(oss.str());
      }
    }
  }

  // 3) At most 1 appearance of (InAlias, OutAlias) pair
  std::vector<std::vector<OutIndex>> seen(nIn);
  for (const auto &crossAlias : m) {
    const auto &x = seen[crossAlias.in().get()];
    if (std::find(x.cbegin(), x.cend(), crossAlias.out()) != x.cend()) {
      std::ostringstream oss;
      oss << "Invalid CrossLinks " << m << ", as the indices of " << x
          << " are duplicated. ";
      throw error(oss.str());
    }
    seen[crossAlias.in().get()].push_back(crossAlias.out());
  }

  // 4) Shape agreement.
  for (const auto &crossAlias : m) {
    if (crossAlias.isModifying() || crossAlias.isAliasing()) {
      const auto inShape  = st.baseState.inShapes[crossAlias.in().get()];
      const auto outShape = st.baseState.outShapes[crossAlias.out().get()];
      if (inShape != outShape) {
        std::ostringstream oss;
        oss << "Incompatible Shapes for CrossLink " << crossAlias
            << ". The input shape at index " << crossAlias.in() << " is "
            << inShape << ", and the output shape at index "
            << crossAlias.out() << " is " << outShape << '.'
            << " CrossAliases which alias must have same "
            << "Shapes for input and output. ";
        throw error(oss.str());
      }
    }
  }

  // 5) We could check the Shapes of the custom Mappers here:

  inIndexIsModified_.resize(nIn, false);
  for (const auto &crossAlias : m) {
    inIndexIsModified_[crossAlias.in_u64()] = crossAlias.isModifying();
  }
}
std::string Multi::typeString() const {
  std::ostringstream oss;
  oss << "Multi(" << mapping() << ')';
  return oss.str();
}

UpBop Multi::clone() const { return mu<Multi>(this); }

bool Multi::modifies(InIndex i) const { return inIndexIsModified_[i.get()]; }

bool Multi::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Multi &>(rhs);
  return mapping() == rhs_.mapping();
}

std::vector<alias::TensorId>
Multi::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {

  std::vector<bool> processed(nOutTensors(), false);
  AliasTensorIds tensorIds(nOutTensors());

  const auto registerOut = [&processed, &tensorIds](OutIndex o,
                                                    alias::TensorId id) {
    processed[o.get()] = true;
    tensorIds[o.get()] = id;
  };

  for (const auto &crossAlias : mapping()) {
    if (crossAlias.isAliasing()) {
      registerOut(crossAlias.out(),
                  g.identity(m.toAliasGraphId(inTensorId(crossAlias.in()))));
    }
  }

  for (uint64_t outIndex = 0; outIndex < nOutTensors(); ++outIndex) {
    if (!processed[outIndex]) {
      registerOut(outIndex, g.allocate(outShape(outIndex), VariableColor));
    }
  }
  return tensorIds;
}

//  ------------  //
// ViewChange1to1 //

/////////
// AliasGate //
/////////
std::vector<alias::TensorId>
AliasGate::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  if (closed()) {
    return {g.allocate(outShape(0), VariableColor)};
  }
  return {g.identity(m.toAliasGraphId(inTensorId(inIndex())))};
}
void AliasGate::close(alias::Graph &g, TensorMap &m) {
  inIndex_ = -1;
  g.toAllocation(m.toAliasGraphId(outTensorId(0)), VariableColor);
}

void AliasGate::openAt(alias::Graph &g, TensorMap &m, InIndex index) {
  if (index.get() >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex (" << index
        << ") in AliasGate::openAt. For AliasGate with " << nInTensors()
        << '.';
    throw error(oss.str());
  }
  inIndex_ = index.get();
  g.toIdentity(m.toAliasGraphId(inTensorId(inIndex_)),
               m.toAliasGraphId(outTensorId(0)));
}

AliasGate::AliasGate(const State &st) : Op(st), inIndex_(-1) {}

AliasGate::AliasGate(const State &st, InIndex i_) : Op(st) {
  if (static_cast<uint64_t>(i_.get()) >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex " << i_ << " in AliasGate constructor. "
        << "Expected value in range [0, " << nInTensors()
        << ") for this AliasGate, which has " << nInTensors() << " inputs. "
        << "Note that closed AliasGatees must be created with the other "
           "(single "
           "input) constructor. ";
    throw error(oss.str());
  }

  inIndex_ = static_cast<int64_t>(i_.get());
}

std::string AliasGate::typeString() const {
  std::ostringstream oss;
  oss << "AliasGate(";
  if (closed()) {
    oss << "closed";
  } else {
    oss << inIndex();
  }
  oss << ')';
  return oss.str();
}

UpBop AliasGate::clone() const { return mu<AliasGate>(this); }

bool AliasGate::inplaceTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const AliasGate &>(rhs);

  // one open, one closed
  if (closed() != rhs_.closed()) {
    return false;
  }

  // both open
  if (open()) {
    return (inIndex() == rhs_.inIndex());
  }

  // both closed
  return true;
}

InIndex AliasGate::inIndex() const {
  if (closed()) {
    throw error("Invalid call, AliasGate::inIndex for closed AliasGate. ");
  }
  return inIndex_;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
