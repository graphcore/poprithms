#include <limits>
#include <sstream>
#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

std::ostream &operator<<(std::ostream &, IsFirst);
std::ostream &operator<<(std::ostream &, IsFinal);

namespace {
void propagate(const Edges &fwd,
               const Edges &bwd,
               std::vector<BitSet> &edgeSet,
               std::vector<uint64_t> &nBefore,
               std::vector<std::array<OpId, 2>> &redundant) {

  auto nOps          = fwd.size();
  auto nBitSetsPerOp = PathMatrix::getNBitSetsPerOp(nOps);

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
      if (isRedundant[bwdIndex]) {
        redundant.push_back({a, b});
      } else {
        record(a, b);
        for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
          edgeSet[b * nBitSetsPerOp + i] |= edgeSet[a * nBitSetsPerOp + i];
        }
      }
    }

    for (uint64_t toSet = 0; toSet < nBitSetsPerOp; ++toSet) {
      nBefore[b] += edgeSet[b * nBitSetsPerOp + toSet].count();
    }
    ++nScheduled;
  }

  if (nScheduled != nOps) {
    throw error("Forward Edges in PathMatrix are not schedulable");
  }
}

template <class T, class U> void removeFromMap(const T &pairs, U &fromToMap) {
  for (const auto &r : pairs) {
    auto from = std::get<0>(r);
    auto to   = std::get<1>(r);
    std::vector<OpId> old;
    old.reserve(fromToMap[from].size() - 1);
    std::swap(old, fromToMap[from]);
    for (auto x : old) {
      if (x != to) {
        fromToMap[from].push_back(x);
      }
    }
  }
}
} // namespace

void PathMatrix::setChains() {
  constexpr ChainId Undef = std::numeric_limits<ChainId>::max();
  opToChainId             = std::vector<ChainId>(nOps, Undef);
  // Guess that the number of Chains is about equal to the number of Ops.
  chainToRootOpId = {};
  chainToRootOpId.reserve(nOps);
  ChainId currentChainId = 0;
  for (OpId opId = 0; opId < nOps; ++opId) {
    if (opToChainId[opId] == Undef) {
      opToChainId[opId] = currentChainId;
      auto tmpId        = opId;
      while (bwd[tmpId].size() == 1 && fwd[bwd[tmpId][0]].size() == 1) {
        tmpId              = bwd[tmpId][0];
        opToChainId[tmpId] = currentChainId;
      }
      chainToRootOpId.push_back(tmpId);
      tmpId = opId;
      while (fwd[tmpId].size() == 1 && bwd[fwd[tmpId][0]].size() == 1) {
        tmpId              = fwd[tmpId][0];
        opToChainId[tmpId] = currentChainId;
      }
      ++currentChainId;
    }
  }
}

void PathMatrix::setChainToUnconstrained() {
  chainIdToUnconstrained.resize(nChains());
  for (ChainId chainId = 0; chainId < nChains(); ++chainId) {
    // Optimization guess on how much to reserve
    if (chainId > 0) {
      chainIdToUnconstrained[chainId].reserve(
          chainIdToUnconstrained[chainId - 1].size());
    }
    auto opId = chainToRootOpId[chainId];
    for (uint64_t i = 0; i < nBitSetsPerOp; ++i) {
      auto index     = opId * nBitSetsPerOp + i;
      BitSet neither = fwdEdgeSet[index] | bwdEdgeSet[index];
      neither.flip();
      if (neither.any()) {
        for (uint64_t shift = 0; shift < BitSetSize; ++shift) {
          auto unconId = i * BitSetSize + shift;
          if (neither[shift] && unconId < nOps && unconId != opId) {
            chainIdToUnconstrained[chainId].push_back(unconId);
          }
        }
      }
    }
  }
}

PathMatrix::PathMatrix(const Edges &_fwd_)
    : nOps(_fwd_.size()), nBitSetsPerOp(getNBitSetsPerOp(nOps)),
      nBitSets(nBitSetsPerOp * nOps), fwd(_fwd_), fwdEdgeSet(nBitSets),
      nFwdBefore(nOps), bwd(nOps), bwdEdgeSet(nBitSets), nBwdBefore(nOps) {

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
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto j : fwd[i]) {
      bwd[j].push_back(i);
    }
  }

  propagate(fwd, bwd, fwdEdgeSet, nFwdBefore, fwdRedundant);
  propagate(bwd, fwd, bwdEdgeSet, nBwdBefore, bwdRedundant);

  removeFromMap(fwdRedundant, fwd);
  removeFromMap(bwdRedundant, bwd);

  setChains();
  setChainToUnconstrained();
}

std::vector<std::tuple<IsFirst, IsFinal>>
PathMatrix::getRelativePositions(const std::vector<OpId> &ids) const {
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

} // namespace pathmatrix
} // namespace schedule

} // namespace poprithms
