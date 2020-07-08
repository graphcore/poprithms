// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/optionalset.hpp>
#include <poprithms/memory/nest/region.hpp>
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

Region Region::permute(const Permutation &p) const {
  return {p.apply(shape().get()), p.apply(setts())};
}

Region Region::reverse(const std::vector<uint64_t> &where) const {
  std::vector<bool> flips(rank_u64(), false);
  for (auto d : where) {
    if (d < rank_u64()) {
      flips[d] = !flips[d];
    } else {
      std::ostringstream oss;
      oss << "Invalid index " << d << " in reverse for " << *this << '.';
      throw error(oss.str());
    }
  }

  std::vector<Sett> flipped;
  flipped.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (flips[d]) {
      flipped.push_back(sett(d).getReverse(0));
    } else {
      flipped.push_back(sett(d));
    }
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

void Region::validateBounds(const std::vector<int64_t> &l,
                            const std::vector<int64_t> &u) const {

  std::ostringstream ss;

  // same rank for lower and upper
  if (l.size() != u.size() || u.size() != rank_u64()) {
    ss << "lower and upper must both be of size "
       << " " << rank_u64() << ". This ia not true for lower=" << l
       << " and upper=" << u << '.';
    throw error(ss.str());
  }

  // lower less than or equal to upper
  for (auto i = 0ul; i < rank_u64(); ++i) {
    if (l[i] > u[i]) {
      ss << "lower bound cannot excede upper bound. "
         << "This for lower=" << l << " and upper=" << u << '.';
      throw error(ss.str());
    }

    if (dim(i) < u[i]) {
      ss << "lower bound cannot excede upper bound. "
         << "This for lower=" << l << " and upper=" << u << '.';
      throw error(ss.str());
    }
  }
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

DisjointSetts DisjointRegions::flattenToSetts() const {
  const auto flatRegs = flatten();
  std::vector<Sett> setts;
  setts.reserve(size());
  for (const auto &reg : flatRegs.get()) {
    setts.push_back(reg.sett(0));
  }
  return DisjointSetts(setts);
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
  if (regs.empty()) {
    ost << "(empty" << regs.shape() << ")";
  } else {
    ost << regs.get();
  }
  return ost;
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

DisjointRegions Region::slice(const std::vector<int64_t> &l,
                              const std::vector<int64_t> &u) const {
  validateBounds(l, u);
  std::vector<Sett> slices;
  slices.reserve(rank_u64());
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    slices.push_back(Sett{{{u[i] - l[i], dim(i) + l[i] - u[i], l[i]}}});
  }
  auto sliced = settSample(Region(shape(), slices));
  if (sliced.size() > 1) {
    std::ostringstream oss;
    oss << "In Region::slice(" << l << ", " << u
        << "), expected 0 or 1 output regions, not " << sliced << '.';
    throw error(oss.str());
  }
  return sliced;
}

} // namespace nest
} // namespace memory
} // namespace poprithms
