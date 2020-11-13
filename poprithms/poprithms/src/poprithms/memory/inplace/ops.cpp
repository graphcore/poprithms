// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

namespace {
template <typename T> std::vector<T> emptyIfOutplaceElseZero(AliasType t) {
  std::vector<T> inds;
  if (!t.isOutplace()) {
    inds.push_back(0);
  }
  return inds;
}

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

namespace {
using UpOp = std::unique_ptr<Op>;
template <typename OP> UpOp mu(const OP *const derived) {
  return std::make_unique<OP>(*derived);
}
} // namespace

// ---------- //
//  NonAlloc  //
// ---------- //
std::vector<alias::TensorId>
NonAlloc::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  if (isOutplace()) {
    AliasTensorIds ids;
    for (uint64_t o = 0; o < nOutTensors(); ++o) {
      ids.push_back(g.allocate(outShape(o), Variable));
    }
    return ids;
  }
  return growInplace(g, m);
}

void NonAlloc::applyOutplaceTo(alias::Graph &g, const TensorMap &m) const {
  g.toAllocation(m.toAliasGraphId(outTensorId(0)), Variable);
}

// -------- //
//  Concat  //
// -------- //
std::string Concat::typeString() const {
  return strcat("Concat(axis=", axis(), ')');
}
bool Concat::typeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Concat &>(rhs);
  return axis() == rhs_.axis();
}
std::vector<alias::TensorId> Concat::growInplace(alias::Graph &g,
                                                 const TensorMap &m) const {
  for (auto x : inTensorIds()) {
    m.toAliasGraphId(x);
  }

  return {g.concat(m.toAliasGraphIds(inTensorIds()), axis())};
}
void Concat::applyInplaceTo(alias::Graph &g,
                            const TensorMap &m,
                            AliasType t) const {
  verifyAllInplace(t);
  g.allocationToConcat(m.toAliasGraphIds(inTensorIds()),
                       axis(),
                       m.toAliasGraphId(outTensorId(0)));
}
std::vector<OutIndex> Concat::outAliasIndicesIf(AliasType t) const {
  return emptyIfOutplaceElseZero<OutIndex>(t);
}
std::vector<InIndex> Concat::inAliasIndicesIf(AliasType t) const {
  if (t.isOutplace()) {
    return {};
  }
  std::vector<InIndex> allInIndices;
  allInIndices.reserve(nInTensors());
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    allInIndices.push_back(InIndex(i));
  }
  return allInIndices;
}
UpOp Concat::clone() const { return mu<Concat>(this); }

// ----------- //
//  UnaryView  //
// ----------- //
std::vector<OutIndex> UnaryView::outAliasIndicesIf(AliasType t) const {
  return emptyIfOutplaceElseZero<OutIndex>(t);
}

std::vector<InIndex> UnaryView::inAliasIndicesIf(AliasType t) const {
  return emptyIfOutplaceElseZero<InIndex>(t);
}

// ------- //
//  Alloc  //
// ------- //
std::string Alloc::typeString() const {
  return strcat("Alloc(color=", color(), ')');
}

bool Alloc::typeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Alloc &>(rhs);
  return color() == rhs_.color();
}

void Alloc::applyOutplaceTo(alias::Graph &, const TensorMap &) const {
  throw error("Alloc cannot grow apply outplace");
}

void Alloc::applyInplaceTo(alias::Graph &,
                           const TensorMap &,
                           AliasType) const {
  throw error("Alloc never changes AliasType, invalid call.");
}

UpOp Alloc::clone() const { return mu<Alloc>(this); }

std::vector<alias::TensorId>
Alloc::typeSpecificGrow(alias::Graph &g, const TensorMap &) const {
  AliasTensorIds ids;
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    ids.push_back(g.allocate(outShape(o), color()));
  }
  return ids;
}

// ------- //
//  Unary  //
// ------- //
std::vector<alias::TensorId> Unary::growInplace(alias::Graph &g,
                                                const TensorMap &m) const {
  return {g.identity(m.toAliasGraphId(inTensorId(0)))};
}

void Unary::applyInplaceTo(alias::Graph &g,
                           const TensorMap &m,
                           AliasType t) const {
  verifyAllInplace(t);
  g.toIdentity(m.toAliasGraphId(inTensorId(0)),
               m.toAliasGraphId(outTensorId(0)));
}
std::unique_ptr<Op> Unary::clone() const { return mu<Unary>(this); }
bool Unary::modifies(InIndex) const { return !isOutplace(); }

std::vector<OutIndex> Unary::outAliasIndicesIf(AliasType t) const {
  return emptyIfOutplaceElseZero<OutIndex>(t);
}
std::vector<InIndex> Unary::inAliasIndicesIf(AliasType t) const {
  return emptyIfOutplaceElseZero<InIndex>(t);
}

// ---------- //
//  Identity  //
// ---------- //
std::vector<alias::TensorId> Identity::growInplace(alias::Graph &g,
                                                   const TensorMap &m) const {
  return {g.identity(m.toAliasGraphId(inTensorId(0)))};
}

void Identity::applyInplaceTo(alias::Graph &g,
                              const TensorMap &m,
                              AliasType t) const {
  verifyAllInplace(t);
  g.toIdentity(m.toAliasGraphId(inTensorId(0)),
               m.toAliasGraphId(outTensorId(0)));
}
UpOp Identity::clone() const { return mu<Identity>(this); }

// ------- //
//  Binary //
// ------- //
// NOTE: if we have in0:(4,)  in1:(1,4,1,1) then either in0 or in1 can be
// inplaced.
std::vector<alias::TensorId> Binary::growInplace(alias::Graph &g,
                                                 const TensorMap &m) const {

  const InIndex inIndex = aliasType().isBinary0() ? 0 : 1;
  return {g.reshape(m.toAliasGraphId(inTensorId(inIndex)), outShape(0))};
}

void Binary::applyInplaceTo(alias::Graph &g,
                            const TensorMap &m,
                            AliasType t) const {
  if (!t.isBinary0() && !t.isBinary1()) {
    std::ostringstream oss;
    oss << "Expected a binary inplace variant in applyInplaceTo, not " << t
        << '.';
    throw error(oss.str());
  }
  const InIndex inIndex = t.isBinary0() ? 0 : 1;
  g.allocationToReshape(m.toAliasGraphId(inTensorId(inIndex)),
                        m.toAliasGraphId(outTensorId(0)));
}
std::unique_ptr<Op> Binary::clone() const { return mu<Binary>(this); }

Binary::Binary(const State &st) : NonAlloc(st) {
  if (!st.aType.isOutplace() && !st.aType.isBinary0() &&
      !st.aType.isBinary1()) {
    std::ostringstream oss;
    oss << "Invalid AliasType in Binary constructor, " << st.aType
        << ". It must be binary0, binary1, or outplace.";
    throw error(oss.str());
  }
}

bool Binary::modifies(InIndex i) const {
  if (aliasType().isBinary0() && i == 0) {
    return true;
  }
  if (aliasType().isBinary1() && i == 1) {
    return true;
  }
  return false;
}
std::vector<OutIndex> Binary::outAliasIndicesIf(AliasType t) const {
  return emptyIfOutplaceElseZero<OutIndex>(t);
}
std::vector<InIndex> Binary::inAliasIndicesIf(AliasType t) const {
  if (t.isBinary0()) {
    return {0};
  }
  if (t.isBinary1()) {
    return {1};
  }
  return {};
}

// ------------ //
//  SettSample  //
// ------------ //
std::string SettSample::typeString() const {
  return strcat("SettSample(region=", region(), ')');
}
bool SettSample::typeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const SettSample &>(rhs);
  return region().equivalent(rhs_.region());
}
std::vector<alias::TensorId>
SettSample::growInplace(alias::Graph &g, const TensorMap &m) const {
  return {g.settsample(m.toAliasGraphId(inTensorId(0)), region())};
}
void SettSample::applyInplaceTo(alias::Graph &g,
                                const TensorMap &m,
                                AliasType t) const {
  verifyAllInplace(t);
  g.allocationToSettsample(m.toAliasGraphId(inTensorId(0)),
                           region(),
                           m.toAliasGraphId(outTensorId(0)));
}

UpOp SettSample::clone() const { return mu<SettSample>(this); }

// ------------ //
//  DimShuffle  //
// ------------ //
std::string DimShuffle::typeString() const {
  return strcat("DimShuffle(permutation=", permutation(), ')');
}
bool DimShuffle::typeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const DimShuffle &>(rhs);
  return permutation() == rhs_.permutation();
}
std::vector<alias::TensorId>
DimShuffle::growInplace(alias::Graph &g, const TensorMap &m) const {
  return {g.dimshuffle(m.toAliasGraphId(inTensorId(0)), permutation())};
}
void DimShuffle::applyInplaceTo(alias::Graph &g,
                                const TensorMap &m,
                                AliasType t) const {
  verifyAllInplace(t);
  g.allocationToDimshuffle(m.toAliasGraphId(inTensorId(0)),
                           permutation(),
                           m.toAliasGraphId(outTensorId(0)));
}
UpOp DimShuffle::clone() const { return mu<DimShuffle>(this); }

// --------- //
//  Reverse  //
// --------- //
std::string Reverse::typeString() const {
  return strcat("Reverse(dimensions=", dimensions().get(), ")");
}
bool Reverse::typeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Reverse &>(rhs);
  return dimensions() == rhs_.dimensions();
}
std::vector<alias::TensorId> Reverse::growInplace(alias::Graph &g,
                                                  const TensorMap &m) const {
  return {g.reverse(m.toAliasGraphId(inTensorId(0)), dimensions().get())};
}
void Reverse::applyInplaceTo(alias::Graph &g,
                             const TensorMap &m,
                             AliasType t) const {
  verifyAllInplace(t);
  g.allocationToReverse(m.toAliasGraphId(inTensorId(0)),
                        dimensions().get(),
                        m.toAliasGraphId(outTensorId(0)));
}
UpOp Reverse::clone() const { return mu<Reverse>(this); }

// --------- //
//  Reshape  //
// --------- //
std::vector<alias::TensorId> Reshape::growInplace(alias::Graph &g,
                                                  const TensorMap &m) const {
  return {g.reshape(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
void Reshape::applyInplaceTo(alias::Graph &g,
                             const TensorMap &m,
                             AliasType t) const {
  verifyAllInplace(t);
  g.allocationToReshape(m.toAliasGraphId(inTensorId(0)),
                        m.toAliasGraphId(outTensorId(0)));
}
UpOp Reshape::clone() const { return mu<Reshape>(this); }

// --------- //
//  Expand   //
// --------- //
std::vector<alias::TensorId> Expand::growInplace(alias::Graph &g,
                                                 const TensorMap &m) const {
  return {g.expand(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
void Expand::applyInplaceTo(alias::Graph &g,
                            const TensorMap &m,
                            AliasType t) const {
  verifyAllInplace(t);
  g.allocationToExpand(m.toAliasGraphId(inTensorId(0)),
                       m.toAliasGraphId(outTensorId(0)));
}
UpOp Expand::clone() const { return mu<Expand>(this); }

// -------------------------- //
// NoneAliasType //
// -------------------------- //
NoneAliasType::NoneAliasType(const State &st) : Op(st) {
  if (st.aType != AliasType::none()) {
    throw error("NoneAliasType, alias type must be none");
  }
}
void NoneAliasType::invalidCall(const std::string &methodName) const {
  std::ostringstream oss;
  oss << "Invalid function class to " << typeString() << ". " << methodName
      << "is not supported for "
      << "NoneAliasType Ops. ";
  throw error(oss.str());
}

void CrossAlias::append(std::ostream &ost) const {
  ost << in() << "->" << out();
  if (isModifying()) {
    ost << "[modifying]";
  } else {
    ost << "[not modifying]";
  }
}

std::ostream &operator<<(std::ostream &ost, const Multi::Mapping &m) {
  ost << '(';
  if (!m.empty()) {
    m[0].append(ost);
  }
  for (uint64_t i = 1; i < m.size(); ++i) {
    ost << ',';
    m[i].append(ost);
  }
  ost << ')';
  return ost;
}

// ------- //
// Multi //
// ------- //
Multi::Multi(const State &st, const Mapping &m)
    : NoneAliasType(st), mapping_(m) {
  const auto nIn = st.inIds.size();
  std::vector<bool> inSeen(nIn, false);
  const auto nOut = st.outShapes.size();
  std::vector<bool> outSeen(nOut, false);
  for (const auto &crossAlias : m) {
    if (crossAlias.in() >= nIn || crossAlias.out() >= nOut ||
        inSeen[crossAlias.in_u64()] || outSeen[crossAlias.out_u64()]) {
      std::ostringstream oss;
      oss << "Invalid Mapping in Multi, " << m;
      oss << ". Number of inputs is " << nIn << ", number of outputs is "
          << nOut << ". "
          << "All input indices must be unique, "
          << "and all output indices must be unique. ";
      throw error(oss.str());
    }
  }

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

UpOp Multi::clone() const { return mu<Multi>(this); }

bool Multi::modifies(InIndex i) const { return inIndexIsModified_[i.get()]; }

bool Multi::typeSpecificEqualTo(const Op &rhs) const {
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
    registerOut(crossAlias.out(),
                g.identity(m.toAliasGraphId(inTensorId(crossAlias.in()))));
  }

  for (uint64_t outIndex = 0; outIndex < nOutTensors(); ++outIndex) {
    if (!processed[outIndex]) {
      registerOut(outIndex, g.allocate(outShape(outIndex)));
    }
  }
  return tensorIds;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
