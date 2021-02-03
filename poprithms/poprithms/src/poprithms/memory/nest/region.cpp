// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/optionalset.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace nest {

namespace {

std::ostream &operator<<(std::ostream &os, const std::vector<int64_t> &foo) {
  util::append(os, foo);
  return os;
}

// Example, if setts = {{a,b}, {c}, {d,e}};
//
// Will return 2 x 1 x 2 DisjointRegions,
//
// {s, {a,c,d}}, {s, {a,c,e}}, {s, {b,c,d}}, {s, {b,c,e}}.
//
DisjointRegions getOuterProduct(const Shape &shape,
                                const std::vector<DisjointSetts> &setts) {

  if (std::any_of(setts.cbegin(), setts.cend(), [](const auto &x) {
        return x.empty();
      })) {
    return DisjointRegions::createEmpty(shape);
  }

  std::vector<std::vector<Sett>> outer{{}};
  std::vector<std::vector<Sett>> prevOuter{};
  outer.reserve(shape.rank_u64());
  prevOuter.reserve(shape.rank_u64());
  for (const auto &atDim : setts) {
    std::swap(prevOuter, outer);
    outer.clear();
    for (const auto &tilDim : prevOuter) {
      for (const auto &p : atDim.get()) {
        auto xCopy = tilDim;
        xCopy.push_back(p);
        outer.push_back(xCopy);
      }
    }
  }
  return DisjointRegions(shape, outer);
}

} // namespace

Region Region::createEmpty(const Shape &sh) {
  return Region(sh,
                std::vector<Sett>(sh.rank_u64(), Sett::createAlwaysOff()));
}

Region Region::createFull(const Shape &sh) {
  return Region(sh, std::vector<Sett>(sh.rank_u64(), Sett::createAlwaysOn()));
}

int64_t Region::nelms(uint64_t d) const { return sett(d).n(dim(d)); }

std::vector<int64_t> Region::nelms() const {
  std::vector<int64_t> ns;
  ns.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    ns.push_back(nelms(d));
  }
  return ns;
}

std::vector<std::vector<int64_t>> Region::getOns() const {
  std::vector<std::vector<int64_t>> ons;
  ons.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    ons.push_back(sett(d).getOns(0, dim(d)));
  }
  return ons;
}

int64_t Region::totalElms() const {
  int64_t n = 1;
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    n *= nelms(d);
    if (n == 0) {
      return 0;
    }
  }
  return n;
}

std::string Region::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

Region Region::dimShuffle(const Permutation &p) const {
  return {p.apply(shape().get()), p.apply(setts())};
}

Region Region::reverse(const std::vector<uint64_t> &where) const {
  std::vector<Sett> flipped = setts();
  for (auto d : shape().getCanonicalReverseIndices(where)) {
    flipped[d] = sett(d).getReverse(0);
  }
  return Region(shape(), flipped);
}

Region Region::expand(const Shape &to) const {
  auto doExpand = shape().numpyWhereToExpand(to);
  auto delta    = to.rank_u64() - rank_u64();
  std::vector<Sett> expandSetts(to.rank_u64(), {{}});
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (!doExpand[i]) {
      expandSetts[i + delta] = sett(i);
    } else {
      // the variables were initialized for this case
    }
  }
  return Region(to, expandSetts);
}

void Region::confirmSameShape(const Region &rhs) const {
  confirmShape(rhs.shape());
}

void Region::confirmShape(const Shape &target) const {
  if (shape() != target) {
    std::ostringstream oss;
    oss << "Failure in Region::confirmShape, where shape=" << shape()
        << " differs from target=" << target;
    throw error(oss.str());
  }
}

OptionalRegion Region::merge(const Region &rhs) const {
  confirmSameShape(rhs);

  uint64_t differingDim = 0;
  bool differingDimFound{false};
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (!sett(d).equivalent(rhs.sett(d))) {
      if (differingDimFound) {
        return OptionalRegion::None();
      }
      differingDimFound = true;
      differingDim      = d;
    }
  }
  if (differingDimFound) {
    auto can1d = Sett::merge(sett(differingDim), rhs.sett(differingDim));
    if (can1d.full()) {
      auto canonSetts          = setts();
      canonSetts[differingDim] = can1d.first();
      Region canonRegion(shape(), canonSetts);
      return OptionalRegion({canonRegion});
    }
  }
  return OptionalRegion::None();
}

// TODO(T24990) accelerate this (make sub-quadratic) with spatial
// data-structure.
DisjointRegions DisjointRegions::intersect(const DisjointRegions &rhs) const {

  std::vector<Region> out;
  for (const auto &a : get()) {
    for (const auto &b : rhs.get()) {
      const auto aInterB = a.intersect(b).get();
      out.insert(out.end(), aInterB.cbegin(), aInterB.cend());
    }
  }

  return DisjointRegions(shape(), out);
}

DisjointRegions Region::intersect(const Region &rhs) const {
  if (empty() || rhs.empty()) {
    return DisjointRegions::createEmpty(shape());
  }
  std::vector<DisjointSetts> partials;
  partials.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    const auto partial = sett(d).intersect(rhs.sett(d));
    if (partial.empty()) {
      return DisjointRegions::createEmpty(shape());
    }
    partials.push_back(std::move(partial));
  }
  return getOuterProduct(shape(), partials);
}

// Example:
//
// If this is the 2-D region ((1,1,0),(2,2,0)):
//
//    xx..xx..xx..
//    ............
//    xx..xx..xx..
//    ............
//
// The complement is the union of
//
//    ............
//    xxxxxxxxxxxx
//    ............
//    xxxxxxxxxxxx
//
// and
//
//    ..xx..xx..xx
//    ............
//    ..xx..xx..xx
//    ............
//
// The general formulation for the complement of A x B x C x ... Z is the
// union of
//
// !A x  1 x  1 x .... x  1
//  A x !B x  1 x .... x  1
//  A x  B x !C x .... x  1
//  A x  B x  C x .... x  1
//  .
//  .
//  .
//  A x  B x  C x .... x !Z.
//
//  where `1' above is the complete set (always on) in a single dimension.
//
//  Using this formulation on the origin 2-D region ((1,1,0),(2,2,0)), we have
//  the union ((1,1,1),()) and ((1,1,0),(2,2,2)), as illustrated.
//

DisjointRegions Region::getComplement() const {
  if (full()) {
    return createEmpty(shape());
  }
  if (empty()) {
    return createFull(shape());
  }

  // implementation of above formulation.
  std::vector<Region> allRegions;
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    std::vector<DisjointSetts> partial;
    for (uint64_t dp = 0; dp < d; ++dp) {
      partial.push_back(sett(dp));
    }
    partial.push_back(sett(d).getComplement());
    for (uint64_t dp = d + 1; dp < rank_u64(); ++dp) {
      partial.push_back({Sett::createAlwaysOn()});
    }
    const auto outer = getOuterProduct(shape(), partial).get();
    allRegions.insert(allRegions.end(), outer.cbegin(), outer.cend());
  }
  return DisjointRegions{shape(), allRegions};
}

DisjointRegions Region::subtract(const Region &rhs) const {
  const auto rhsCompl = rhs.getComplement();
  std::vector<Region> intersection;
  for (const auto &complElm : rhsCompl.get()) {
    const auto inter = intersect(complElm).get();
    intersection.insert(intersection.end(), inter.cbegin(), inter.cend());
  }
  return DisjointRegions(shape(), intersection);
}

Region::Region(const Shape &sh_, const std::vector<Sett> &se_) : shape_(sh_) {

  if (sh_.rank_u64() != se_.size()) {
    std::ostringstream oss;
    oss << "In Region constructor, with shape=" << sh_
        << ", and setts=" << se_ << ". "
        << "Expected shape and setts to have the same size. ";
    throw error(oss.str());
  }

  setts_.reserve(sh_.rank_u64());
  for (uint64_t i = 0; i < sh_.rank_u64(); ++i) {
    if (sh_.dim(i) == 0) {
      setts_.push_back(Sett::createAlwaysOn());
    } else {
      auto periodic = se_[i].adjustedPrepend({sh_.dim(i), 0, 0});
      setts_.push_back(periodic);
    }
  }
}

bool Region::empty() const {
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (nelms(d) == 0) {
      return true;
    }
  }
  return false;
}

bool Region::full() const { return totalElms() == shape().nelms(); }

bool Region::contains(const Region &rhs) const {
  confirmSameShape(rhs);
  if (rhs.empty()) {
    return true;
  }
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (!sett(d).contains(rhs.sett(d))) {
      return false;
    }
  }
  return true;
}

bool Region::equivalent(const Region &rhs) const {
  return equivalent({*this}, {rhs});
}

bool Region::equivalent(const DisjointRegions &lhs,
                        const DisjointRegions &rhs) {

  if (lhs.shape() != rhs.shape()) {
    return false;
  }

  const auto lhsFlat = lhs.flattenToSetts();
  const auto rhsFlat = rhs.flattenToSetts();
  return lhsFlat.equivalent(rhsFlat);
}

std::ostream &operator<<(std::ostream &ost, const Region &r) {
  r.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const DisjointRegions &regs) {
  regs.append(ost);
  return ost;
}

void DisjointRegions::append(std::ostream &ost) const {
  if (full()) {
    ost << "(full" << shape() << ")";
  } else if (empty()) {
    ost << "(empty" << shape() << ")";
  } else {
    ost << "(shape=" << shape() << ",(";
    for (uint64_t i = 0; i < regs_.size(); ++i) {
      ost << at(i).setts();
      if (i + 1 < regs_.size()) {
        ost << ',';
      }
    }
    ost << "))";
  }
}

void DisjointRegions::appendBitwise(std::ostream &ost) const {
  std::vector<std::string> regStrings;
  for (const auto &r : get()) {
    regStrings.push_back(r.getBitwiseString());
  }
  poprithms::util::append(ost, regStrings);
}

std::string Region::getBitwiseString() const {
  std::ostringstream oss;
  appendBitwise(oss);
  return oss.str();
}

std::ostream &operator<<(std::ostream &ost, const std::vector<Region> &regs) {
  poprithms::util::append(ost, regs);
  return ost;
}

void Region::append(std::ostream &ost) const {
  ost << "(shape=" << shape() << ",setts=";
  poprithms::util::append(ost, setts());
  ost << ")";
}

namespace {

void appendBoolOns(std::ostream &ost, const std::vector<bool> &vs) {
  util::append(ost, vs);
}

} // namespace

void Region::appendBitwise(std::ostream &ost) const {
  if (rank_u64() == 0) {
    ost << "()";
    return;
  }
  ost << '(';
  appendBoolOns(ost, sett(0).getBoolOns(0, dim(0)));
  for (uint64_t i = 1; i < rank_u64(); ++i) {
    ost << "  x  ";
    appendBoolOns(ost, sett(i).getBoolOns(0, dim(i)));
  }
  ost << ')';
}

DisjointRegions Region::settSample(const Region &where) const {
  confirmSameShape(where);
  const auto outShape = where.nelms();
  if (std::any_of(outShape.cbegin(), outShape.cend(), [](const auto &v) {
        return v == 0;
      })) {
    return DisjointRegions::createEmpty(outShape);
  }
  std::vector<DisjointSetts> partials;
  partials.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    partials.push_back(sett(d).sampleAt(where.sett(d)));
  }
  return getOuterProduct(where.nelms(), partials);
}

DisjointRegions DisjointRegions::settSample(const Region &where) const {
  std::vector<Region> outs;
  for (const Region &r : get()) {
    const auto partOuts = r.settSample(where).get();
    outs.insert(outs.end(), partOuts.cbegin(), partOuts.cend());
  }
  return DisjointRegions(where.nelms(), outs);
}

DisjointRegions Region::settFillInto(const Region &scaffold) const {
  confirmShape(scaffold.nelms());
  std::vector<DisjointSetts> partials;
  partials.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    partials.push_back(scaffold.sett(d).fillWith(sett(d)));
  }
  return getOuterProduct(scaffold.shape(), partials);
}

DisjointRegions Region::settFillWith(const Region &ink) const {
  ink.confirmShape(nelms());
  std::vector<DisjointSetts> partials;
  partials.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    partials.push_back(sett(d).fillWith(ink.sett(d)));
  }
  return getOuterProduct(shape(), partials);
}

DisjointRegions Region::reshape(const Shape &to) const {
  if (shape().nelms() != to.nelms()) {
    std::ostringstream oss;
    oss << "Invalid call Region::reshape(" << to << "). Cannot reshape from "
        << shape() << " to " << to
        << ", as the number of elements is not conserved.";
    throw error(oss.str());
  }
  const auto flatRegion      = flatten();
  auto unFlatDisjointRegions = flatRegion.unflatten(to);
  return unFlatDisjointRegions;
}

Region Region::flatten() const {
  const auto strides = shape().getRowMajorStrides();
  std::vector<Stripe> stripes;
  for (auto d = 0UL; d < rank_u64(); ++d) {
    for (auto stp : sett(d).getStripes()) {
      stripes.push_back(stp.getScaled(strides[d]));
    }
  }
  return Region(Shape({shape().nelms()}), {Sett(stripes)});
}

DisjointRegions Region::unflatten(const Shape &to) const {

  if (rank_u64() != 1) {
    std::ostringstream oss;
    oss << "Cannot call Region::unflatten for " << *this
        << " as it is not rank-1.";
    throw error(oss.str());
  }

  if (to.rank_u64() == 0) {
    if (totalElms() != 1) {
      std::ostringstream oss;
      oss << "Cannot call Region::unflatten(to=" << to << ") for " << *this
          << ", as only regions with 1 element can be unflattened to rank-0.";
      throw error(oss.str());
    }
    const Shape scalar{};
    return DisjointRegions(scalar,
                           std::vector<Region>(1, Region(scalar, {})));
  }

  const auto N = to.nelms();
  if (N != dim(0)) {
    std::ostringstream oss;
    oss << "Target shape in Region::unflatten (" << to
        << ") is not valid, it has an "
        << "incompatible number "
        << "of elements with " << shape();
    throw error(oss.str());
  }

  std::vector<std::vector<Sett>> toProcess{setts()};
  decltype(toProcess) nextToProcess;

  for (uint64_t d = 0; d < to.rank_u64() - 1; ++d) {
    for (const auto &vp : toProcess) {
      const auto &pref = vp[0];
      auto unflattened = pref.unflatten(to.dim(to.rank_u64() - d - 1));
      for (const auto &unflat : unflattened) {
        std::vector<Sett> nxt;
        nxt.push_back(std::get<0>(unflat));
        nxt.push_back(std::get<1>(unflat));
        nxt.insert(nxt.end(), std::next(vp.cbegin()), vp.cend());
        nextToProcess.push_back(nxt);
      }
    }
    std::swap(toProcess, nextToProcess);
    nextToProcess.clear();
  }

  std::vector<Region> nDisjointRegions;
  nDisjointRegions.reserve(toProcess.size());
  for (const auto &vp : toProcess) {
    nDisjointRegions.push_back({to, vp});
  }

  return DisjointRegions(to, nDisjointRegions);
}

bool Region::disjoint(const Region &rhs) const {
  auto flat    = flatten();
  auto rhsFlat = rhs.flatten();
  return flat.sett(0).disjoint(rhsFlat.sett(0));
}

Region Region::fromBounds(const Shape &sh,
                          const std::vector<int64_t> &l,
                          const std::vector<int64_t> &u) {

  sh.assertSliceBoundsAreValid(l, u);
  std::vector<Sett> slices;
  slices.reserve(sh.rank_u64());
  for (uint64_t i = 0; i < sh.rank_u64(); ++i) {
    slices.push_back(Sett{{{u[i] - l[i], sh.dim(i) + l[i] - u[i], l[i]}}});
  }

  return Region(sh, slices);
}

Region
Region::fromBounds(const Shape &s, Dimension d, uint64_t l, uint64_t u) {

  // Easy check for validity of d, l, u:
  s.slice(d, l, u);

  std::vector<Sett> setts(s.rank_u64(), Sett::createAlwaysOn());
  int64_t on     = u - l;
  int64_t off    = s.dim(d.get()) - on;
  int64_t phase  = l;
  setts[d.get()] = {{{on, off, phase}}};
  return Region(s, setts);
}

// TODO(T32863) use Dimension instead of uint64_t.
Region Region::fromStripe(const Shape &sh,
                          uint64_t dimension,
                          const Stripe &stripe0) {
  if (dimension >= sh.rank_u64()) {
    std::ostringstream oss;
    oss << "Call to Region::fromStripe invalid for shape=" << sh
        << " and dimension=" << dimension;
    throw error(oss.str());
  }
  std::vector<Sett> setts(sh.rank_u64(), {{{}}});
  setts[dimension] = Sett({stripe0});
  return Region(sh, setts);
}

Region Region::slice(const std::vector<int64_t> &l,
                     const std::vector<int64_t> &u) const {

  const auto sampleRegion = fromBounds(shape(), l, u);
  auto sliced             = settSample(sampleRegion);
  if (sliced.size() > 1) {
    std::ostringstream oss;
    oss << "In Region::slice(" << l << ", " << u
        << "), expected 0 or 1 output regions, not " << sliced << '.';
    throw error(oss.str());
  }

  if (sliced.size() == 0) {
    return Region::createEmpty(shape().slice(l, u));
  }
  return sliced.at(0);
}

Region Region::reduce(const Shape &outShape) const {

  shape().assertCanReduceTo(outShape);
  const auto outRank   = outShape.rank_u64();
  const auto deltaRank = rank_u64() - outRank;

  std::vector<nest::Sett> nxtSetts(outRank, {{}});
  for (uint64_t d = 0; d < outShape.rank_u64(); ++d) {
    if (outShape.dim(d) != 1) {
      nxtSetts[d] = sett(d + deltaRank);
    }
  }
  return Region{outShape, nxtSetts};
}

// DisjointRegions //
// =============== //

DisjointRegions DisjointRegions::reduce(const Shape &outShape) const {
  // Example:
  //
  // 11..11..11..   and
  // .11..11..11.   should reduce to,
  // 111.111.111.   .
  //
  // i.e. the reduction is the intersection of the reduced Regions.

  DisjointRegions outRegs(outShape, std::vector<Region>{});
  for (const auto &toReduce : get()) {
    const auto reduction = toReduce.reduce(outShape);

    // Add the novel contribution of this Region's reduction make to the
    // union.
    const auto toInsert = DisjointRegions(reduction).subtract(outRegs);
    for (auto r : toInsert.get()) {
      outRegs.insert(r);
    }
  }
  return outRegs;
}

bool DisjointRegions::contains(const DisjointRegions &rhs) const {
  const auto nElmsThis = totalElms();

  // Check that each Region in rhs is contained, individually:
  for (const auto &r : rhs.get()) {
    const auto diff0 = subtract(DisjointRegions(r));
    if (nElmsThis - diff0.totalElms() != r.totalElms()) {
      return false;
    }
  }
  return true;
}

DisjointRegions DisjointRegions::subtract(const DisjointRegions &rhs) const {

  // Iteratively subtract the Regions in rhs from this object's Regions.
  // Initialize with complete Regions, nothing subtracted yet:
  std::vector<Region> reduced = get();

  std::vector<Region> nxt;

  for (const auto &toSubtract : rhs.get()) {
    for (const auto &r : reduced) {
      const auto smaller = r.subtract(toSubtract).get();
      nxt.insert(nxt.end(), smaller.cbegin(), smaller.cend());
    }
    std::swap(reduced, nxt);
    nxt.clear();
  }
  return DisjointRegions(shape(), reduced);
}

int64_t DisjointRegions::totalElms() const {
  const auto counts = nelms();
  return std::accumulate(counts.cbegin(), counts.cend(), 0LL);
}

void DisjointRegions::insert(const Region &r) {
  if (r.shape() != shape()) {
    std::ostringstream oss;
    oss << "Incompatible Shape in DisjointRegions::insert. "
        << "The DisjointRegions has shape " << shape()
        << ", the Region being inserted has shape " << r.shape();
    throw error(oss.str());
  }
  regs_.push_back(r);
}

DisjointRegions
DisjointRegions::slice(const std::vector<int64_t> &lower,
                       const std::vector<int64_t> &upper) const {
  std::vector<Region> allOutRegions;
  for (const auto &reg : get()) {
    const auto inReg = reg.slice(lower, upper);
    if (!inReg.empty()) {
      allOutRegions.push_back(inReg);
    }
  }
  return DisjointRegions(shape().slice(lower, upper), allOutRegions);
}

DisjointRegions DisjointRegions::settFillInto(const Region &scaffold) const {
  std::vector<Region> oRegs;
  oRegs.reserve(size());
  for (const auto &reg : get()) {
    auto inRegs = reg.settFillInto(scaffold);
    for (const auto &inReg : inRegs.get()) {
      if (!inReg.empty()) {
        oRegs.push_back(inReg);
      }
    }
  }
  return DisjointRegions(scaffold.shape(), oRegs);
}

DisjointRegions
DisjointRegions::reverse(const std::vector<uint64_t> &dimensions) const {
  std::vector<Region> oRegs;
  oRegs.reserve(size());
  for (const auto &reg : get()) {
    oRegs.push_back(reg.reverse(dimensions));
  }
  return DisjointRegions(shape(), oRegs);
}

DisjointRegions DisjointRegions::dimShuffle(const Permutation &p) const {
  std::vector<Region> oRegs;
  oRegs.reserve(size());
  for (const auto &reg : get()) {
    const auto inReg = reg.dimShuffle(p);
    oRegs.push_back(reg.dimShuffle(p));
  }
  return DisjointRegions(p.apply(shape().get()), oRegs);
}

DisjointRegions DisjointRegions::reshape(const Shape &s) const {
  std::vector<Region> oRegs;
  oRegs.reserve(size());
  for (const auto &reg : get()) {
    auto inRegs = reg.reshape(s);
    for (const auto &inReg : inRegs.get()) {
      if (!inReg.empty()) {
        oRegs.push_back(inReg);
      }
    }
  }
  return DisjointRegions(s, oRegs);
}

bool DisjointRegions::disjoint(const DisjointRegions &rhs) const {
  if (rhs.shape() != shape()) {
    std::ostringstream oss;
    oss << "Error in DisjointRegions::disjoint, where this has Shape "
        << shape() << " and rhs has Shape " << rhs.shape();
    throw error(oss.str());
  }

  // using triangle inequality on set sizes:
  if (shape().nelms() < totalElms() + rhs.totalElms()) {
    return false;
  }

  for (const auto &reg0 : get()) {
    for (const auto &reg1 : rhs.get()) {
      if (!reg0.disjoint(reg1)) {
        return false;
      }
    }
  }

  return true;
}

DisjointRegions::DisjointRegions(const Shape &s,
                                 const std::vector<Region> &rs)
    : sh_(s), regs_(rs) {

  for (const auto &reg : rs) {
    if (reg.shape() != s) {
      std::ostringstream oss;
      oss << "Expected shape and regions to agree in DisjointRegions "
          << "constructor. This is not true for regions=" << rs
          << " and shape=" << s;
      throw error(oss.str());
    }
  }
}

bool DisjointRegions::isValid() const {

  if (empty()) {
    return true;
  }

  for (uint64_t d = 0; d < size(); ++d) {
    if (at(d).shape() != shape()) {
      return false;
    }
    for (uint64_t d2 = 0; d2 < d; ++d2) {
      if (!at(d).disjoint(at(d2))) {
        return false;
      }
    }
  }
  return true;
}

void DisjointRegions::confirmValid() const {
  if (!isValid()) {
    std::ostringstream oss;
    oss << *this
        << " is not valid, failure in DisjoingRegions::confirmValid.";
    throw error(oss.str());
  }
}

std::vector<Region>
DisjointRegions::regsFromSetts(const Shape &sh,
                               const std::vector<std::vector<Sett>> &se) {
  std::vector<Region> rs;
  rs.reserve(se.size());
  for (const auto &x : se) {
    rs.push_back({sh, x});
  }
  return rs;
}

std::vector<int64_t> DisjointRegions::nelms() const {
  std::vector<int64_t> ns;
  ns.reserve(size());
  for (const auto &reg : get()) {
    ns.push_back(reg.totalElms());
  }
  return ns;
}

DisjointRegions DisjointRegions::flatten() const {
  std::vector<Region> flat;
  flat.reserve(size());
  for (const auto &reg : get()) {
    flat.push_back(reg.flatten());
  }
  return DisjointRegions(Shape({shape().nelms()}), flat);
}

DisjointRegions DisjointRegions::expand(const Shape &s) const {
  std::vector<Region> expanded;
  expanded.reserve(size());
  for (const auto &reg : get()) {
    expanded.push_back(reg.expand(s));
  }
  return DisjointRegions(s, expanded);
}

DisjointSetts DisjointRegions::flattenToSetts() const {
  const auto flatRegs = flatten();
  std::vector<Sett> setts;
  setts.reserve(size());
  for (const auto &reg : flatRegs.get()) {
    setts.push_back(reg.sett(0));
  }
  return DisjointSetts(setts);
}

namespace {
std::vector<Sett> settsFromStrides(const Strides &strides) {
  std::vector<Sett> setts;
  setts.reserve(strides.size());
  for (auto stride : strides.get()) {
    setts.push_back({{{1, static_cast<int64_t>(stride) - 1, 0}}});
  }
  return setts;
}
} // namespace

Region Region::fromStrides(const Shape &shape, const Strides &strides) {
  return Region(shape, settsFromStrides(strides));
}

Region Region::fromStride(const Shape &shape, Stride s, Dimension d) {
  return Region::fromStripe(shape, d.get_i64(), {1, s.get_i64() - 1, 0});
}

DisjointRegions DisjointRegions::getComplement() const {
  return DisjointRegions::createFull(shape()).subtract(*this);
}

} // namespace nest
} // namespace memory
} // namespace poprithms
