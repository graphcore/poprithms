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
std::string Concat::typeString() const {
  return strcat("Concat(axis=", axis(), ')');
}
bool Concat::typeSpecificEqualTo(const Op &rhs) const {
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
UpOp Concat::clone() const { return mu<Concat>(this); }

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

UpOp Alloc::clone() const { return mu<Alloc>(this); }

std::vector<alias::TensorId>
Alloc::typeSpecificGrow(alias::Graph &g, const TensorMap &) const {
  AliasTensorIds ids;
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    ids.push_back(g.allocate(outShape(o), color()));
  }
  return ids;
}

std::vector<alias::TensorId> Mux::typeSpecificGrow(alias::Graph &g,
                                                   const TensorMap &m) const {
  if (closed()) {
    return {g.allocate(outShape(0), VariableColor)};
  }
  return {g.identity(m.toAliasGraphId(inTensorId(inIndex())))};
}

// ------- //
//  Unary  //
// ------- //
std::vector<alias::TensorId>
UnaryModifier::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.identity(m.toAliasGraphId(inTensorId(0)))};
}
std::unique_ptr<Op> UnaryModifier::clone() const {
  return mu<UnaryModifier>(this);
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
SettSample::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.settsample(m.toAliasGraphId(inTensorId(0)), region())};
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
DimShuffle::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.dimshuffle(m.toAliasGraphId(inTensorId(0)), permutation())};
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
std::vector<alias::TensorId>
Reverse::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.reverse(m.toAliasGraphId(inTensorId(0)), dimensions().get())};
}
UpOp Reverse::clone() const { return mu<Reverse>(this); }

// --------- //
//  Reshape  //
// --------- //
std::vector<alias::TensorId>
Reshape::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.reshape(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
UpOp Reshape::clone() const { return mu<Reshape>(this); }

// --------- //
//  Expand   //
// --------- //
std::vector<alias::TensorId>
Expand::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.expand(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}

UpOp Expand::clone() const { return mu<Expand>(this); }

// ------- //
// Multi //
// ------- //
Multi::Multi(const State &st, const CrossAliases &m) : Op(st), mapping_(m) {
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
      registerOut(outIndex, g.allocate(outShape(outIndex), VariableColor));
    }
  }
  return tensorIds;
}

void Mux::close(alias::Graph &g, TensorMap &m) {
  inIndex_ = -1;
  g.toAllocation(m.toAliasGraphId(outTensorId(0)), VariableColor);
}

void Mux::openAt(alias::Graph &g, TensorMap &m, InIndex index) {
  if (index.get() >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex (" << index << ") in Mux::openAt. For Mux with "
        << nInTensors() << '.';
    throw error(oss.str());
  }
  inIndex_ = index.get();
  g.toIdentity(m.toAliasGraphId(inTensorId(inIndex_)),
               m.toAliasGraphId(outTensorId(0)));
}

Mux::Mux(const State &st) : Op(st), inIndex_(-1) {}

Mux::Mux(const State &st, InIndex i_) : Op(st) {
  if (static_cast<uint64_t>(i_.get()) >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex " << i_ << " in Mux constructor. "
        << "Expected value in range [0, " << nInTensors()
        << ") for this Mux, which has " << nInTensors() << " inputs. "
        << "Note that closed Muxes must be created with the other (single "
           "input) constructor. ";
    throw error(oss.str());
  }

  inIndex_ = static_cast<int64_t>(i_.get());
}

std::string Mux::typeString() const {
  std::ostringstream oss;
  oss << "Mux(";
  if (closed()) {
    oss << "closed";
  } else {
    oss << inIndex();
  }
  oss << ')';
  return oss.str();
}

UpOp Mux::clone() const { return mu<Mux>(this); }

bool Mux::typeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const Mux &>(rhs);
  return closed() == rhs_.closed() || (inIndex() == rhs_.inIndex());
}

InIndex Mux::inIndex() const {
  if (closed()) {
    throw error("Invalid call, Mux::inIndex for closed Mux. ");
  }
  return inIndex_;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
