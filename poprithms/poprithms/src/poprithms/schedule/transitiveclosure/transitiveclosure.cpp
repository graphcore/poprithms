// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <limits>
#include <sstream>

#include <poprithms/schedule/transitiveclosure/error.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

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

void propagate(const Edges &fwd,
               const Edges &bwd,
               std::vector<BitSet> &edgeSet) {

  auto nOps          = fwd.size();
  auto nBitSetsPerOp = TransitiveClosure::getNBitSetsPerOp(nOps);

  auto isRecordered = [&edgeSet, nBitSetsPerOp](OpId from, OpId to) {
    auto index = to * nBitSetsPerOp + from / BitSetSize;
    auto shift = from % BitSetSize;
    return edgeSet[index][shift];
  };

  auto record = [&edgeSet, nBitSetsPerOp](OpId from, OpId to) {
    auto index            = to * nBitSetsPerOp + from / BitSetSize;
    auto shift            = from % BitSetSize;
    edgeSet[index][shift] = true;
  };

  uint64_t nScheduled{0};
  std::vector<OpId> outstanding;
  outstanding.reserve(nOps);
  std::vector<OpId> ready;
  for (OpId i = 0; i < nOps; ++i) {
    outstanding.push_back(bwd[i].size());
    if (outstanding[i] == 0) {
      ready.push_back(i);
    }
  }

  while (!ready.empty()) {

    //       c
    //      /
    // a - b - c`
    //   /
    // a`
    //
    // b signals to the c's that it is scheduled,
    // and then b sets its edge-set from the a's edge-sets.
    //
    // 1) signal to cs.
    OpId b = ready.back();
    ready.pop_back();
    for (auto c : fwd[b]) {
      --outstanding[c];
      if (outstanding[c] == 0) {
        ready.push_back(c);
      }
    }

    // 2) set edge-set from a's, and record which a's are redundant
    // performance note: int is better than bool for the task at hand.
    std::vector<int32_t> isRedundant(bwd[b].size(), false);
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

  if (nScheduled != nOps) {
    throw error("Forward Edges in TransitiveClosure are not schedulable");
  }
}

} // namespace

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
      nBitSets(nBitSetsPerOp * nOps), fwdEdgeSet(nBitSets),
      bwdEdgeSet(nBitSets) {

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
  std::vector<std::vector<OpId>> bwd(nOps);
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto j : fwd[i]) {
      bwd[j].push_back(i);
    }
  }

  propagate(fwd, bwd, fwdEdgeSet);
  propagate(bwd, fwd, bwdEdgeSet);
}

std::vector<BitSet> TransitiveClosure::getBits(const Filters &filters) const {

  std::vector<BitSet> soln(nBitSetsPerOp);

  for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
    if (nOps - i * BitSetSize >= BitSetSize) {
      soln[i].set(); // set to true
    } else {
      soln[i].reset();
      for (uint64_t j = 0; j < nOps % BitSetSize; ++j) {
        soln[i].set(j, true);
      }
    }
    for (const auto &f : filters) {
      auto type  = std::get<0>(f);
      auto opId  = std::get<1>(f);
      auto index = opId * nBitSetsPerOp + i;

      if (type == IsFirst::Maybe) {
        BitSet neither = fwdEdgeSet[index] | bwdEdgeSet[index];
        neither.flip();
        if (i == (opId / BitSetSize)) {
          neither[opId % BitSetSize] = false;
        }
        soln[i] &= neither;
      }

      else if (type == IsFirst::Yes) {
        soln[i] &= fwdEdgeSet[index];
      }

      else {
        soln[i] &= bwdEdgeSet[index];
      }
    }
  }

  return soln;
}

std::vector<OpId> TransitiveClosure::get(const Filters &filters) const {
  auto soln = getBits(filters);
  std::vector<OpId> uCon;

  for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
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

uint64_t TransitiveClosure::n(const Filters &filters) const {
  auto soln = getBits(filters);
  uint64_t c{0};
  for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
    c += soln[i].count();
  }
  return c;
}

bool TransitiveClosure::same(IsFirst r, const std::vector<OpId> &ids) const {
  if (ids.size() < 2) {
    return true;
  }
  auto soln0 = getBits({{r, ids[0]}});
  for (auto iter = std::next(ids.cbegin()); iter != ids.cend();
       std::advance(iter, 1)) {
    auto soln1 = getBits({{r, *iter}});
    for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
      if (soln0[i] != soln1[i]) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::tuple<IsFirst, IsFinal>>
TransitiveClosure::getExtremumStatuses(const std::vector<OpId> &ids) const {
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
TransitiveClosure::getExtremumStatus(OpId a,
                                     const std::vector<OpId> &subset) const {

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
  for (auto x : get({{IsFirst::Maybe, id}})) {
    if (earliest(x) < e) {
      return false;
    }
  }
  return true;
}

uint64_t TransitiveClosure::earliest(OpId id) const {
  return n({{IsFirst::Yes, id}});
}

uint64_t TransitiveClosure::latest(OpId id) const {
  return nOps - n({{IsFirst::No, id}}) - 1;
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
                                         std::vector<BitSet> &edgeSet) {

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

} // namespace transitiveclosure
} // namespace schedule

} // namespace poprithms
