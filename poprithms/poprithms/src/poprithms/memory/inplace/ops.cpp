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

DisjointRegions Concat::outRegions(const DisjointRegions &inRegs,
                                   InIndex i,
                                   OutIndex o) const {
  verify(i, o, "outRegions");
  return inRegs.settFillInto(
      Region::fromBounds(outShape(0), getLowerSlice(i), getUpperSlice(i)));
}

DisjointRegions Concat::inRegions(const DisjointRegions &outRegs,
                                  InIndex i,
                                  OutIndex o) const {
  verify(i, o, "inRegions");
  return outRegs.slice(getLowerSlice(i), getUpperSlice(i));
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
DisjointRegions
Alloc::outRegions(const DisjointRegions &, InIndex, OutIndex) const {
  throw error("No Alloc::outRegions implemented, as no valid InIndex");
}
DisjointRegions
Alloc::inRegions(const DisjointRegions &, InIndex, OutIndex) const {
  throw error("No Alloc::inRegions implemented, as no valid InIndex");
}
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
DisjointRegions SettSample::inRegs(const DisjointRegions &out) const {
  return out.settFillInto(region());
}
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
DisjointRegions DimShuffle::inRegs(const DisjointRegions &out) const {
  return out.permute(permutation().inverse());
}
DisjointRegions DimShuffle::outRegs(const DisjointRegions &inRegs) const {
  return inRegs.permute(permutation());
}
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
DisjointRegions Reverse::inRegs(const DisjointRegions &out) const {
  return out.reverse(dimensions().get());
}
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
DisjointRegions Reshape::inRegs(const DisjointRegions &out) const {
  return out.reshape(inShape(0));
}
std::vector<alias::TensorId>
Reshape::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.reshape(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
UpOp Reshape::clone() const { return mu<Reshape>(this); }

Reshape::Reshape(const State &st) : ViewChange1to1(st) {
  if (st.inShapes.size() != 1 || st.outShapes.size() != 1) {
    throw error("Invalid reshape, expected 1 input and 1 output");
  }

  if (st.outShapes[0].nelms_u64() != st.inShapes[0].nelms_u64()) {
    std::ostringstream oss;
    oss << "Invalid reshape, number of elements changes. "
        << "Cannot reshape from " << st.inShapes[0] << " to "
        << st.outShapes[0] << ". ";
    throw error(oss.str());
  }
}

// --------- //
//  Expand   //
// --------- //
DisjointRegions Expand::inRegs(const DisjointRegions &out) const {
  return out.reduce(inShape(0));
}
DisjointRegions Expand::outRegs(const DisjointRegions &inRegs) const {
  return inRegs.expand(inShape(0));
}
std::vector<alias::TensorId>
Expand::typeSpecificGrow(alias::Graph &g, const TensorMap &m) const {
  return {g.expand(m.toAliasGraphId(inTensorId(0)), outShape(0))};
}
UpOp Expand::clone() const { return mu<Expand>(this); }

// ----- //
// Multi //
// ----- //
DisjointRegions
Multi::outRegions(const DisjointRegions &rs, InIndex i, OutIndex o) const {
  verify(i, o, "outRegions");
  for (const auto &crossAlias : mapping()) {
    if (crossAlias.in() == i && crossAlias.out() == o) {
      return crossAlias.fwd(rs);
    }
  }
  return DisjointRegions::createEmpty(outShape(o));
}

DisjointRegions
Multi::inRegions(const DisjointRegions &rs, InIndex i, OutIndex o) const {
  verify(i, o, "inRegions");
  for (const auto &crossAlias : mapping()) {
    if (crossAlias.in() == i && crossAlias.out() == o) {
      return crossAlias.bwd(rs);
    }
  }
  return DisjointRegions::createEmpty(inShape(i));
}
Multi::Multi(const State &st, const CrossLinks &m) : Op(st), mapping_(m) {
  const auto nIn  = st.inIds.size();
  const auto nOut = st.outShapes.size();

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
      const auto inShape  = st.inShapes[crossAlias.in().get()];
      const auto outShape = st.outShapes[crossAlias.out().get()];
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
//  ------------  //

DisjointRegions ViewChange1to1::outRegions(const DisjointRegions &inRegs,
                                           InIndex i,
                                           OutIndex o) const {
  verify(i, o, "outRegions");
  return outRegs(inRegs);
}

DisjointRegions ViewChange1to1::inRegions(const DisjointRegions &outRegs,
                                          InIndex i,
                                          OutIndex o) const {
  verify(i, o, "inRegions");
  return inRegs(outRegs);
}

/////////
// Mux //
/////////
DisjointRegions
Mux::outRegions(const DisjointRegions &inRegs, InIndex i, OutIndex o) const {
  verify(i, o, "outRegions");
  return inRegs.expand(outShape(0));
}

DisjointRegions
Mux::inRegions(const DisjointRegions &outRegs, InIndex i, OutIndex o) const {
  verify(i, o, "inRegions");
  return outRegs.reduce(inShape(0));
}
std::vector<alias::TensorId> Mux::typeSpecificGrow(alias::Graph &g,
                                                   const TensorMap &m) const {
  if (closed()) {
    return {g.allocate(outShape(0), VariableColor)};
  }
  return {g.identity(m.toAliasGraphId(inTensorId(inIndex())))};
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
