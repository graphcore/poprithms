#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

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
} // namespace

PathMatrix::PathMatrix(const Edges &_fwd_)
    : nOps(_fwd_.size()), nBitSetsPerOp(getNBitSetsPerOp(nOps)),
      nBitSets(nBitSetsPerOp * nOps), fwd(_fwd_), fwdEdgeSet(nBitSets),
      nFwdBefore(nOps), bwd(nOps), bwdEdgeSet(nBitSets), nBwdBefore(nOps) {

  // setting bwd
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto j : fwd[i]) {
      bwd[j].push_back(i);
    }
  }

  propagate(fwd, bwd, fwdEdgeSet, nFwdBefore, fwdRedundant);
  propagate(bwd, fwd, bwdEdgeSet, nBwdBefore, bwdRedundant);
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

} // namespace pathmatrix
} // namespace schedule
} // namespace poprithms
