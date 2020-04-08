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

// TODO(T16486) clarify the use of Chains in PathMatrix
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

  // Returns true if and only if (iff) there exists
  // at least 1 schedule with a before b, and
  // at least 1 schedule with b before a.
  bool unconstrained(OpId a, OpId b) const {
    return !constrained(a, b) && !constrained(b, a);
  }

  using Filter  = std::tuple<IsFirst, OpId>;
  using Filters = std::vector<Filter>;
  // Filters are used in get (below) with the following semantics
  //  {IsFirst::Yes, a}
  //      will be true for all b s.t. a is before b in all schedules,
  //  {IsFirst::Maybe, a}
  //      will be true for all b s.t. a is before b in at least 1 schedule,
  //      and a is after b in at least 1 schedule,
  //  {IsFirst::No, a}
  //      will be true for all b s.t. a is after b in all schedules.

  // Get the intersection of all Filters.
  //   Example:
  //   get({{IsFirst::Yes, a}, {IsFirst::Maybe, b}, {IsFirst::No, c}})
  //   returns the set of all Ops which are
  //   - always after a (as the first filter is "a is IsFirst::Yes")
  //   - sometimes before a, sometimes after b
  //   - always before c.
  std::vector<OpId> get(const Filters &) const;

  // The size of the set returned by get(.)
  uint64_t n(const Filters &) const;

  // return true if get({{x, y}}) is the same for all y in ys.
  bool same(IsFirst x, const std::vector<OpId> &ys) const;

  // convenience functions
  std::vector<OpId> getUnconstrained(OpId id) const {
    return get({{IsFirst::Maybe, id}});
  }
  std::vector<OpId> getPost(OpId id) const {
    return get({{IsFirst::No, id}});
  }
  std::vector<OpId> getUnconstrainedPost(OpId a, OpId b) const {
    return get({{IsFirst::Maybe, a}, {IsFirst::No, b}});
  }
  bool sameUnconstrained(OpId a, OpId b) const {
    return same(IsFirst::Maybe, {a, b});
  }
  uint64_t nPostPost(OpId a, OpId b) const {
    return n({{IsFirst::No, a}, {IsFirst::No, b}});
  }

  uint64_t nOps_u64() const { return nOps; }
  int64_t nOps_i64() const { return static_cast<int64_t>(nOps); }

  static uint64_t getNBitSetsPerOp(uint64_t nOps) {
    return nOps / BitSetSize + (nOps % BitSetSize != 0);
  }

  // for each Op \in subOps, what can be said about its position in a schedule
  // relative to each of the other Ops in subOps? For example, if Op a appears
  // before all b \in subOps (where b != a) in all schedules, then "a" has
  // IsFirst::Yes returned from this function
  std::vector<std::tuple<IsFirst, IsFinal>>
  getRelativePositions(const std::vector<OpId> &subOps) const;

  std::vector<std::array<OpId, 2>>
  getFlattenedRedundants(const Edges &) const;

  Edges getRedundants(const Edges &) const;

  bool asEarlyAsAllUnconstrained(OpId id) const;
  uint64_t earliest(OpId id) const;
  uint64_t latest(OpId id) const;

private:
  uint64_t nOps;
  uint64_t nBitSetsPerOp;
  uint64_t nBitSets;

  std::vector<BitSet> fwdEdgeSet;
  std::vector<BitSet> bwdEdgeSet;

  std::vector<BitSet> getBits(const Filters &) const;

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
