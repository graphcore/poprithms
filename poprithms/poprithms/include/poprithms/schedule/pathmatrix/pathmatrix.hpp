#ifndef POPRITHMS_SCHEDULE_EDGEMAP_EDGEMAP_HPP
#define POPRITHMS_SCHEDULE_EDGEMAP_EDGEMAP_HPP

#include <array>
#include <bitset>
#include <tuple>
#include <vector>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

// The number of bits stored per std::bitset,
// see bitset_performance_0.cpp for an run to choose this value.
// std::bitset is a good data-structure here, as it has a very fast
// count() method, probably using x86's popcnt
static constexpr uint64_t BitSetSize = 512;

using BitSet  = std::bitset<BitSetSize>;
using OpId    = uint64_t;
using SchedId = uint64_t;
using Edges   = std::vector<std::vector<OpId>>;

// TODO(jn) PathMatrix currently used Chains to store more compactly the
// unconstrained sets. This is poor design - it should use Chains to store
// everything or nothing more compactly. I think there should be 2 classes:
// one which has no concept of Chains, and stores every field for every Op
// separately, and another which has an instance of the first, and maps chains
// to Ops
using ChainId = uint32_t;

enum class IsFirst { No = 0, Maybe, Yes };
enum class IsFinal { No = 0, Maybe, Yes };

// A class for compactly storing all dependencies between Nodes (Ops) in a
// DAG. Queries for implicit topological constraints between any 2 Ops are
// performed in O(1) time. Memory consumption is O(nOps^2) and object
// construction time is O(nOps * nEdges). The implemenation of the class is
// careful to keep the constants in these complexities low.
class PathMatrix {

public:
  PathMatrix(const Edges &forwardEdges);

  // Returns true if there exist no schedules with "to" before "from"
  bool constrained(OpId from, OpId to) const {
    auto index = to * nBitSetsPerOp + from / BitSetSize;
    auto shift = from % BitSetSize;
    return fwdEdgeSet[index][shift];
  }

  // Returns true iff there exists
  // at least 1 schedule with a before b, and
  // at least 1 schedule with b before a.
  bool unconstrained(OpId a, OpId b) const {
    return !constrained(a, b) && !constrained(b, a);
  }

  const std::vector<OpId> &getUnconstrained(OpId id) const {
    return chainIdToUnconstrained[opToChainId[id]];
  }

  // The lowest SchedId that "a" has over all schedules
  SchedId earliest(OpId a) const { return nFwdBefore[a]; }

  // The highest SchedId that "a" has over all schedules
  SchedId latest(OpId a) const { return nOps_u64() - nBwdBefore[a] - 1; }

  uint64_t nOps_u64() const { return nOps; }
  uint64_t nOps_i64() const { return static_cast<int64_t>(nOps); }

  // The set of forward edges passed to the constructor which are redundant.
  // That is, all edges which if removed would not change the total number of
  // schedules
  const auto &getFwdRedundant() const { return fwdRedundant; }

  // The same edges as getFwdRedundant(), but reversed
  const auto &getBwdRedundant() const { return bwdRedundant; }

  static uint64_t getNBitSetsPerOp(uint64_t nOps) {
    return nOps / BitSetSize + (nOps % BitSetSize != 0);
  }

  // Forward edges, with redundant ones removed
  const Edges &getFwd() const { return fwd; }

  // Backward edges, with redundant ones removed
  const Edges &getBwd() const { return bwd; }

  // for each Op \in subOps, what can be said about its position in a schedule
  // relative to each of the other Ops in subOps? For example, if Op a appears
  // before all b \in subOps (where b != a) in all schedules, then "a" has
  // IsFirst::Yes returned from this function
  std::vector<std::tuple<IsFirst, IsFinal>>
  getRelativePositions(const std::vector<OpId> &subOps) const;

  uint64_t nChains() const { return chainToRootOpId.size(); }

private:
  uint64_t nOps;
  uint64_t nBitSetsPerOp;
  uint64_t nBitSets;

  Edges fwd;
  std::vector<BitSet> fwdEdgeSet;
  std::vector<uint64_t> nFwdBefore;
  std::vector<std::array<OpId, 2>> fwdRedundant;

  Edges bwd;
  std::vector<BitSet> bwdEdgeSet;
  std::vector<uint64_t> nBwdBefore;
  std::vector<std::array<OpId, 2>> bwdRedundant;

  std::vector<ChainId> opToChainId;
  std::vector<OpId> chainToRootOpId;
  void setChains();

  std::vector<std::vector<OpId>> chainIdToUnconstrained;
  void setChainToUnconstrained();

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
  // A PathMatrix is O(nOps^2) in memory. Each of fwdEdgeSet and
  // bwdEdgeSet store nOps^2 + O(1) bits, and record forward and backard
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
  // faster using the transposed layout, and so it IS stored.
  //
  //
  // Example:
  //
  //       X0
  //      / \
  //     X1  X2
  //      \ /
  //       X3
  //         \
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
};

std::ostream &operator<<(std::ostream &, schedule::pathmatrix::IsFirst);
std::ostream &operator<<(std::ostream &, schedule::pathmatrix::IsFinal);

} // namespace pathmatrix
} // namespace schedule

} // namespace poprithms

#endif
