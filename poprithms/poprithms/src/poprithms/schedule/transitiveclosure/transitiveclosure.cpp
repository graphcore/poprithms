// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <limits>
#include <random>
#include <sstream>

#include <schedule/transitiveclosure/error.hpp>

#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

// Diagram:
//          from
//
//        **** ****
//        **** ****
//  to    **** ****
//        **** ****
//        **** ****
//        **** ****
//        **** ****
//        **** ****
//
// A TransitiveClosure is O(nOps^2) in memory. Each of fwdEdgeSet and
// bwdEdgeSet store nOps*(nOps+ O(1)) bits, they record forward and backard
// constraints respectively.
//
// In the diagram above, BitSetSize is 4 and nOps is 8. Each * in the
// diagram is a constraint between 2 Ops, and will either be on or off.
//
// The majority of time spent in the construction is in bitwise addition of
// 2 rows, and summation over columns.
//
// Note that bwdEdgeSet is the transpose of fwdEdgeSet, and so is not
// required to be stored. However, certain operations are significantly
// faster using the transposed layout, and so it is stored.
//
// Example:
//
//       X0
//      / \..
//     X1  X2
//      \ /..
//       X3
//         \..
//          X4
//
//  has fwdEdgeSet:
//
//       from
//       01234
//     0 00000
//     1 10000
//  to 2 10000
//     3 11100
//     4 11110
//

std::ostream &operator<<(std::ostream &, IsFirst);
std::ostream &operator<<(std::ostream &, IsFinal);

namespace {

void verifyOpAddresses(const Edges &edges, uint64_t nOps) {
  for (const auto &evs : edges) {
    for (auto e : evs) {
      if (e >= nOps) {
        std::ostringstream oss;
        oss << "Invalid edge end, " << e << ", with only " << nOps << " Ops.";
        throw error(oss.str());
      }
    }
  }
}

void propagate(const Edges &fwd, const Edges &bwd, BitSets &edgeSet) {

  const auto nOps          = fwd.size();
  const auto nBitSetsPerOp = TransitiveClosure::getNBitSetsPerOp(nOps);

  /**
   * Return true if the transitive closure has a 'true' bit denoting that
   * there's a path from #from to #to.
   * */
  auto isRecordered = [&edgeSet, nBitSetsPerOp](OpId from, OpId to) {
    const auto index = to * nBitSetsPerOp + from / BitSetSize;
    const auto shift = from % BitSetSize;
    return edgeSet[index][shift];
  };

  /**
   * Set a 'true' bit in the transitiveclosure to denote that there's a path
   * from #from to #to.
   * */
  auto record = [&edgeSet, nBitSetsPerOp](OpId from, OpId to) {
    const auto index      = to * nBitSetsPerOp + from / BitSetSize;
    const auto shift      = from % BitSetSize;
    edgeSet[index][shift] = true;
  };

  /**
   * We process the Ops in forward topological order.
   * */
  uint64_t nScheduled{0};
  OpIds outstanding;
  outstanding.reserve(nOps);
  OpIds ready;
  for (OpId i = 0; i < nOps; ++i) {
    outstanding.push_back(bwd[i].size());
    if (outstanding[i] == 0) {
      ready.push_back(i);
    }
  }

  while (!ready.empty()) {

    //
    // a ---+          +----> c
    //      |          |
    //      +--- b ----+
    //      |          |
    // a` --+          +----> c'
    //
    // b is popped off the back of ready, as it has all of its input edges
    // satisfied:

    OpId b = ready.back();
    ready.pop_back();

    // Then,
    //
    // (1) b signals to its outputs, the c's above, that it is scheduled, and
    // then
    // (2) b sets its edge-set from the a's edge-sets.

    // (1) signalling to c's.
    for (auto c : fwd[b]) {
      --outstanding[c];
      if (outstanding[c] == 0) {
        ready.push_back(c);
      }
    }

    // 2) set edge-set from a's, and record which a's are redundant
    // performance note: int is better than bool for the task at hand.
    std::vector<int32_t> isRedundant(bwd[b].size(), false);

    //
    //
    // a0  ---+
    //  ^     |
    //  |     v
    // a1  ---+---> b
    //  |     |
    //  ^     ^
    //  |     |
    // a2  ---+
    //
    // The edge a2 -> b is redundant in the above diagram.
    // The edge a1 -> b is also redundant.

    // For all i which have a direct edge to b, if there is a j with a direct
    // edge to b, and there is a path from i to j, the path from i to j is
    // redundant.
    //
    // Note that the order in which the edges are processed does not matter.
    for (uint64_t i = 0; i < bwd[b].size(); ++i) {
      uint64_t j = 0;
      while (j < bwd[b].size() && !isRedundant[i]) {
        if (i == j) {
          ++j;
        } else if (isRedundant[j]) {
          ++j;
        } else if (isRecordered(bwd[b][i], bwd[b][j])) {
          isRedundant[i] = true;
        } else {
          ++j;
        }
      }
    }

    for (uint64_t bwdIndex = 0; bwdIndex < bwd[b].size(); ++bwdIndex) {
      auto a = bwd[b][bwdIndex];
      if (!isRedundant[bwdIndex]) {
        record(a, b);
        for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
          edgeSet[b * nBitSetsPerOp + i] |= edgeSet[a * nBitSetsPerOp + i];
        }
      }
    }

    ++nScheduled;
  }

  // As there is no context information about the nodes here, we don't print
  // the strongly connected components.
  if (nScheduled != nOps) {
    std::ostringstream oss;
    oss << "Forward Edges in TransitiveClosure are not schedulable, "
        << "there is a cycle in this Graph. ";
    throw error(oss.str());
  }
}

} // namespace

BitSets TransitiveClosure::getAllTrue(uint64_t nOps) {
  const auto nBitSets = getNBitSetsPerOp(nOps);
  BitSets soln(nBitSets);
  for (uint64_t i = 0; i < nBitSets; ++i) {
    if (nOps - i * BitSetSize >= BitSetSize) {
      soln[i].set(); // set to true
    } else {
      // in the final bitset, we ensure that any trailing out-of-bounds bits
      // remain false.
      soln[i].reset(); // set to false
      for (uint64_t j = 0; j < nOps % BitSetSize; ++j) {
        soln[i].set(j, true);
      }
    }
  }
  return soln;
}

BitSets TransitiveClosure::getAllFalse(uint64_t nOps) {
  const auto nBitSets = getNBitSetsPerOp(nOps);
  BitSets soln(nBitSets);
  for (uint64_t i = 0; i < nBitSets; ++i) {
    soln[0].reset(); // set to false
  }
  return soln;
}

uint64_t TransitiveClosure::n(const BitSets &bs) {
  uint64_t c{0};
  for (const auto &b : bs) {
    c += b.count();
  }
  return c;
}

Edges TransitiveClosure::getRedundants(const Edges &edges) const {
  Edges revEdges(edges.size());
  for (OpId from = 0; from < edges.size(); ++from) {
    for (auto to : edges[from]) {
      revEdges[to].push_back(from);
    }
  }

  Edges redundants(edges.size());
  for (OpId from = 0; from < edges.size(); ++from) {
    for (OpId to : edges[from]) {
      for (auto toPrime : revEdges[to]) {
        if (constrained(from, toPrime)) {
          redundants[from].push_back(to);
          break;
        }
      }
    }
  }
  return redundants;
}

std::vector<std::array<OpId, 2>>
TransitiveClosure::getFlattenedRedundants(const Edges &edges) const {
  Edges redEdges = getRedundants(edges);
  std::vector<std::array<OpId, 2>> redundants;
  for (OpId from = 0; from < redEdges.size(); ++from) {
    for (auto to : redEdges[from]) {
      redundants.push_back({from, to});
    }
  }
  return redundants;
}

TransitiveClosure::TransitiveClosure(const Edges &fwd)
    : nOps(fwd.size()), nBitSetsPerOp(getNBitSetsPerOp(nOps)),
      fwdEdgeSet(nBitSetsPerOp * nOps), bwdEdgeSet(nBitSetsPerOp * nOps) {

  bidirectionalPropagate(fwd);
}

void TransitiveClosure::bidirectionalPropagate(const Edges &fwd) {

  for (const auto &evs : fwd) {
    for (auto e : evs) {
      if (e >= nOps) {
        std::ostringstream oss;
        oss << "Invalid edge end, " << e << ", with only " << nOps << " Ops.";
        throw error(oss.str());
      }
    }
  }

  // setting bwd
  std::vector<OpIds> bwd(nOps);
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto j : fwd[i]) {
      bwd[j].push_back(i);
    }
  }

  propagate(fwd, bwd, fwdEdgeSet);
  propagate(bwd, fwd, bwdEdgeSet);
}

bool TransitiveClosure::operator==(const TransitiveClosure &x) const {
  return fwdEdgeSet == x.fwdEdgeSet && bwdEdgeSet == x.bwdEdgeSet;
}

template <typename Combiner>
BitSets TransitiveClosure::bitSetCombine(Filters filters,
                                         Combiner &&c) const {

  BitSets soln = c.init();

  // It might be possible to simplify/reduce the filters before applying them.
  // For example ((IsFirst::No, a), (IsFirst::No, b)) is simply (IsFirst::No,
  // b) if b is always after a, and we're taking an intersection.
  // TODO(T43562)
  filters = c.simplifiedFilters(filters, *this);

  // We shuffle the filters, so that early termination becomes more likely.
  //
  // Note that std::shuffle is not guarenteed to be the same across operating
  // systems and platforms, but that does not matter for our purpose here,
  // because the returned BitSets do not depend on the randomness.
  std::mt19937 g(1011);
  std::shuffle(filters.begin(), filters.end(), g);

  //  The first time at which we check for early termination.
  uint64_t earlyTerminationCheckTime{4};

  for (uint64_t fi = 0; fi < filters.size(); ++fi) {
    const auto &f = filters[fi];

    if (fi == earlyTerminationCheckTime) {
      earlyTerminationCheckTime *= 2;
      if (c.isFixedPoint(n(soln))) {
        return soln;
      }
    }

    auto type = std::get<0>(f);
    auto opId = std::get<1>(f);

    for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
      auto index = opId * nBitSetsPerOp + i;

      if (type == IsFirst::Maybe) {
        c.combine(soln[i], getIsFirstMaybe(opId, i));
      }

      else if (type == IsFirst::Yes) {
        c.combine(soln[i], fwdEdgeSet[index]);
      }

      else {
        c.combine(soln[i], bwdEdgeSet[index]);
      }
    }
  }

  return soln;
}

class TransitiveClosure::Intersecter {
public:
  Intersecter(uint64_t nOps) : nOps_(nOps) {}
  BitSets init() const { return TransitiveClosure::getAllTrue(nOps_); }
  const uint64_t nOps_;
  void combine(BitSet &a, const BitSet &b) const { a &= b; }

  // If every bit is false, then any intersection will not change that.
  bool isFixedPoint(uint64_t nTrue) const { return nTrue == 0; }

  // TODO(T43562)
  TransitiveClosure::Filters
  simplifiedFilters(const Filters &fs, const TransitiveClosure &) const {
    return fs;
  }
};

class TransitiveClosure::Unioner {
public:
  Unioner(uint64_t nOps) : nOps_(nOps) {}
  BitSets init() const { return TransitiveClosure::getAllFalse(nOps_); }
  const uint64_t nOps_;
  void combine(BitSet &a, const BitSet &b) const { a |= b; }
  bool isFixedPoint(uint64_t nTrue) const { return nTrue == nOps_; }

  // TODO(T43562)
  TransitiveClosure::Filters
  simplifiedFilters(const Filters &fs, const TransitiveClosure &) const {
    return fs;
  }
};

BitSets TransitiveClosure::bitSetIntersection(const Filters &filters) const {
  return bitSetCombine(filters, Intersecter(nOps_u64()));
}

BitSets TransitiveClosure::bitSetUnion(const Filters &filters) const {
  return bitSetCombine(filters, Unioner(nOps_u64()));
}

template <typename Combiner>
BitSets
TransitiveClosure::bitSetCombine(const std::vector<BitSets> &toCombine,
                                 Combiner &&combiner) const {
  auto combined = combiner.init();
  for (const auto &toMerge : toCombine) {
    for (uint64_t j = 0; j < toMerge.size(); ++j) {
      combiner.combine(combined[j], toMerge[j]);
    }
  }
  return combined;
}

BitSets TransitiveClosure::bitSetIntersection(
    const std::vector<BitSets> &bitsets) const {
  auto x = bitSetCombine(bitsets, Intersecter(nOps_u64()));
  return x;
}

BitSets
TransitiveClosure::bitSetUnion(const std::vector<BitSets> &bitsets) const {
  return bitSetCombine(bitsets, Unioner(nOps_u64()));
}

BitSet TransitiveClosure::getIsFirstMaybe(OpId opId,
                                          uint64_t bitsetIndex) const {

  auto index     = opId * nBitSetsPerOp + bitsetIndex;
  BitSet neither = fwdEdgeSet[index] | bwdEdgeSet[index];
  neither.flip();
  if (bitsetIndex == (opId / BitSetSize)) {
    neither[opId % BitSetSize] = false;
  }
  return neither;
}

TransitiveClosure::DurationBound
TransitiveClosure::getDurationBound(const OpIds &ops) const {

  if (ops.size() < 2) {
    return {ops.size(), ops.size() + 1};
  }

  // Which subset of 'ops' might be before all others in the 'ops'. And,
  // Which subset of 'ops' might be after all others in the 'ops'.
  const auto extremumStatuses = getExtremumStatuses(ops);
  OpIds mightBeFirsts;
  OpIds mightBeFinals;
  uint64_t nOnEdge{0};
  for (uint64_t i = 0; i < ops.size(); ++i) {

    const auto stat       = extremumStatuses[i];
    const bool maybeFirst = (std::get<0>(stat) != IsFirst::No);
    const bool maybeFinal = (std::get<1>(stat) != IsFinal::No);

    if (maybeFirst || maybeFinal) {
      ++nOnEdge;
      if (maybeFirst) {
        mightBeFirsts.push_back(ops[i]);
      }
      if (maybeFinal) {
        mightBeFinals.push_back(ops[i]);
      }
    }
  }

  const auto nMightBeFirsts = mightBeFirsts.size();
  const auto nMightBeFinals = mightBeFinals.size();

  // How many Ops are definitely before all Ops in 'ops'?
  transitiveclosure::TransitiveClosure::Filters beforeFirsts;
  beforeFirsts.reserve(nMightBeFirsts);
  for (auto mightBeFirst : mightBeFirsts) {
    beforeFirsts.push_back({IsFirst::Yes, mightBeFirst});
  }
  const auto nBefore = nIntersection(beforeFirsts);

  // How many Ops are definitely after all Ops in 'ops'?
  transitiveclosure::TransitiveClosure::Filters afterFinals;
  afterFinals.reserve(nMightBeFinals);
  for (auto mightBeFinal : mightBeFinals) {
    afterFinals.push_back({IsFirst::No, mightBeFinal});
  }
  const auto nAfter = nIntersection(afterFinals);

  // How may Ops are
  // 1) definitely after at least one op in mightBeFirsts, and
  // 2) definitely before at least one op in mightBeFinals?
  transitiveclosure::TransitiveClosure::Filters afterOneFirst;
  afterOneFirst.reserve(nMightBeFirsts);
  for (uint64_t i = 0; i < nMightBeFirsts; ++i) {
    afterOneFirst.push_back({IsFirst::No, mightBeFirsts[i]});
  }
  const auto definitelyAfterOne = bitSetUnion(afterOneFirst);

  transitiveclosure::TransitiveClosure::Filters beforeOneFinal;
  beforeOneFinal.reserve(nMightBeFinals);
  for (uint64_t i = 0; i < nMightBeFinals; ++i) {
    beforeOneFinal.push_back({IsFirst::Yes, mightBeFinals[i]});
  }
  const auto definitelyBeforeOne = bitSetUnion(beforeOneFinal);

  const auto nInbetween =
      n(bitSetIntersection({{definitelyAfterOne, definitelyBeforeOne}}));

  // We want to know if, for all schedules, finalInSchedule('ops') -
  // firstInSchedule('ops) is fixed. Are the Ops which might be in range
  // [first, final) sometimes, but not others? It's definitely none of the
  // following, if so:
  //
  // 1) definitely before all 'mightBeFirsts'.
  //
  // 2) definitely after all 'mightBeFinals'.
  //
  // 3) definitely not before all of 'mightBeFirsts' and definitely not after
  //    all 'mightBeFinals'
  //
  // 4) in 'mightBeFirsts' or 'mightBeFinals'.
  //
  // If it's not one of the above 4 categories, then it migt be in the
  // interval [first, final), and it might be out.
  const auto accountedFor = nBefore + nAfter + nInbetween + nOnEdge;

  if (accountedFor > nOps_u64()) {
    throw error("Logic error in getDurationBound. Sums of sizes of mutually "
                "exclusive subsets cannot exceed size of parent set. ");
  }

  const auto l = nOnEdge + nInbetween;
  const auto u = l + 1 + nOps_u64() - accountedFor;
  return {l, u};
}

OpIds TransitiveClosure::opIntersection(const Filters &filters) const {
  return opIds(bitSetIntersection(filters));
}

OpIds TransitiveClosure::opUnion(const Filters &filters) const {
  return opIds(bitSetUnion(filters));
}

OpIds TransitiveClosure::opIds(const BitSets &soln) {
  OpIds uCon;
  for (uint64_t i = 0; i < soln.size(); ++i) {
    if (soln[i].any()) {
      for (uint64_t shift = 0; shift < BitSetSize; ++shift) {
        auto id = i * BitSetSize + shift;
        if (soln[i][shift]) {
          uCon.push_back(id);
        }
      }
    }
  }

  return uCon;
}

std::ostream &operator<<(std::ostream &ost, const BitSets &bs) {
  auto ids = TransitiveClosure::opIds(bs);
  poprithms::util::append(ost, ids);
  return ost;
}

uint64_t TransitiveClosure::nIntersection(const Filters &filters) const {
  return n(bitSetIntersection(filters));
}

uint64_t TransitiveClosure::nUnion(const Filters &filters) const {
  return n(bitSetUnion(filters));
}

bool TransitiveClosure::same(IsFirst r, const OpIds &ids) const {
  if (ids.size() < 2) {
    return true;
  }
  auto soln0 = get({r, ids[0]});
  for (auto iter = std::next(ids.cbegin()); iter != ids.cend();
       std::advance(iter, 1)) {
    auto soln1 = get({r, *iter});
    for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
      if (soln0[i] != soln1[i]) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::tuple<IsFirst, IsFinal>>
TransitiveClosure::getExtremumStatuses(const OpIds &ids) const {
  std::vector<std::tuple<IsFirst, IsFinal>> rps;
  rps.resize(ids.size(), {IsFirst::Yes, IsFinal::Yes});
  for (uint64_t idIndex = 0; idIndex < ids.size(); ++idIndex) {
    const auto id0 = ids[idIndex];
    auto &isFirst  = std::get<0>(rps[idIndex]);
    auto &isFinal  = std::get<1>(rps[idIndex]);
    for (const auto &id1 : ids) {
      if (id1 == id0) {
      } else if (constrained(id0, id1)) {
        isFinal = IsFinal::No;
      } else if (constrained(id1, id0)) {
        isFirst = IsFirst::No;
      } else {
        if (isFirst != IsFirst::No) {
          isFirst = IsFirst::Maybe;
        }
        if (isFinal != IsFinal::No) {
          isFinal = IsFinal::Maybe;
        }
      }
    }
  }
  return rps;
}

std::tuple<IsFirst, IsFinal>
TransitiveClosure::getExtremumStatus(OpId a, const OpIds &subset) const {

  // For a to be IsFirst::Yes, constrained(a, b) must be true for ALL b. For a
  // to be IsFirst::No, constrained(b, a) must be true for at least 1 b.
  auto isFirst = IsFirst::Yes;
  for (auto b : subset) {

    if (a != b) {

      // There's a b which is not constrained to be before a, so a is not
      // IsFirst::Yes. We can't break from the loop with IsFirst::Maybe
      // though, as we might still "drop" from IsFirst::Maybe to IsFirst::No.
      if (!constrained(a, b)) {
        isFirst = IsFirst::Maybe;
      }

      // There's a b which is before a, so therefore a is IsFirst::No.
      if (constrained(b, a)) {
        isFirst = IsFirst::No;
        break;
      }
    }
  }

  // The logic for IsFinal is the same as for IsFirst.
  auto isFinal = IsFinal::Yes;
  for (auto b : subset) {
    if (a != b) {
      if (!constrained(b, a)) {
        isFinal = IsFinal::Maybe;
      }
      if (constrained(a, b)) {
        isFinal = IsFinal::No;
        break;
      }
    }
  }

  return {isFirst, isFinal};
}

// TODO : there may be a more efficient way to implement this
bool TransitiveClosure::asEarlyAsAllUnconstrained(OpId id) const {
  const auto e = earliest(id);
  for (auto x : get({IsFirst::Maybe, id})) {
    if (earliest(x) < e) {
      return false;
    }
  }
  return true;
}

std::ostream &operator<<(std::ostream &os, IsFirst isFirst) {
  switch (isFirst) {
  case (IsFirst::No): {
    os << "IsFirst::No";
    break;
  }
  case (IsFirst::Maybe): {
    os << "IsFirst::Maybe";
    break;
  }
  case (IsFirst::Yes): {
    os << "IsFirst::Yes";
    break;
  }
  }
  return os;
}

void TransitiveClosure::update(const Edges &newEdges) {
  verifyOpAddresses(newEdges, nOps);
  for (OpId from = 0; from < newEdges.size(); ++from) {
    for (auto to : newEdges[from]) {
      insertConstraint(from, to, fwdEdgeSet);
      insertConstraint(to, from, bwdEdgeSet);
    }
  }
}

void TransitiveClosure::insertConstraint(OpId from,
                                         OpId to,
                                         BitSets &edgeSet) {

  auto isRecordered = [&edgeSet, this](OpId f, OpId t) {
    auto index = t * nBitSetsPerOp + f / BitSetSize;
    auto shift = f % BitSetSize;
    return edgeSet[index][shift];
  };

  auto record = [&edgeSet, this](OpId f, OpId t) {
    auto index            = t * nBitSetsPerOp + f / BitSetSize;
    auto shift            = f % BitSetSize;
    edgeSet[index][shift] = true;
    for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
      edgeSet[t * nBitSetsPerOp + i] |= edgeSet[f * nBitSetsPerOp + i];
    }
  };

  if (!isRecordered(from, to)) {
    record(from, to);
    for (OpId postTo = 0; postTo < nOps; ++postTo) {
      if (isRecordered(to, postTo) && !isRecordered(from, postTo)) {
        record(from, postTo);
      }
    }
  }
}

std::ostream &operator<<(std::ostream &os, IsFinal isFinal) {
  switch (isFinal) {
  case (IsFinal::No): {
    os << "IsFinal::No";
    break;
  }
  case (IsFinal::Maybe): {
    os << "IsFinal::Maybe";
    break;
  }
  case (IsFinal::Yes): {
    os << "IsFinal::Yes";
    break;
  }
  }
  return os;
}

std::ostream &operator<<(std::ostream &ost,
                         const std::tuple<IsFirst, IsFinal> &p) {
  ost << '(' << std::get<0>(p) << ", " << std::get<1>(p) << ')';
  return ost;
}

bool TransitiveClosure::unconstrainedWithAtLeastOne(
    OpId opId,
    uint64_t bitSetIndex) const {

  // We handle the final bitset group, where not all bits correspond to ops,
  // as a special case. This is the 'slow' approach of checking each bit
  // separately.
  if ((bitSetIndex + 1) * BitSetSize > nOps) {
    for (uint64_t i = bitSetIndex * BitSetSize; i < nOps; ++i) {
      if (unconstrainedInBothDirections(opId, i)) {
        return true;
      }
    }
    return false;
  }

  // The non-final bitset, where all bits correspond to ops:
  else {
    auto index     = opId * nBitSetsPerOp + bitSetIndex;
    BitSet neither = fwdEdgeSet[index] | bwdEdgeSet[index];
    neither.flip();
    if (neither.any()) {
      return true;
    }
    return false;
  }
}

std::ostream &operator<<(std::ostream &ost,
                         const TransitiveClosure::DurationBound &db) {
  ost << "[shortest=" << db.low << ",longest=" << db.high << ")";
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const TransitiveClosure &tc) {
  for (uint64_t row = 0; row < tc.nOps_u64(); ++row) {
    ost << "\n  ";
    for (uint64_t col = 0; col < tc.nOps_u64(); ++col) {
      ost << tc.constrained(row, col);
    }
    ost << "    (" << row << " is before)";
  }
  return ost;
}

uint64_t TransitiveClosure::getNBitSets(OpId) const { return nBitSetsPerOp; }

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
