// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_EDGEMAP_EDGEMAP_HPP
#define POPRITHMS_SCHEDULE_EDGEMAP_EDGEMAP_HPP

#include <array>
#include <bitset>
#include <tuple>
#include <vector>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

// The number of bits stored per std::bitset, see bitset_performance_0.cpp for
// an run to choose this value. std::bitset is a good data-type to use to
// store the transitive closure bits, as it is compact, and it has a fast
// count() method, probably compiled x86's popcnt instruction.
static constexpr uint64_t BitSetSize = 512;
using BitSet                         = std::bitset<BitSetSize>;

using OpId  = uint64_t;
using Edges = std::vector<std::vector<OpId>>;

enum class IsFirst { No = 0, Maybe, Yes };
enum class IsFinal { No = 0, Maybe, Yes };

std::ostream &operator<<(std::ostream &,
                         const std::tuple<IsFirst, IsFinal> &);

/**
 * A class for compactly storing all indirect topological constraints between
 * Nodes (Ops) in a DAG. Example: suppose constraints between a,b and c are
 * a -> b
 * b -> c.
 * Then the transitive closure will contain a->c as this is an indirect
 * constraint. This is exactly the mathematical structure desribed at:
 *
 * https://en.wikipedia.org/wiki/Transitive_closure
 */
class TransitiveClosure {

public:
  /**
   * Construct from an explicit set of DAG Edges.
   */
  TransitiveClosure(const Edges &forwardEdges);

  /**
   * Insert additional DAG edges. Note that it is much faster to call the
   * constructor with the full set of edges than to incrementally call update.
   */
  void update(const Edges &newEdges);

  /**
   * Return true if there is a constraint (implicit or explicit) "from before
   * to", or from->to. In other words, return true if there exist no schedules
   * with "to" before "from". This query is performed on O(1) time.
   * */
  bool constrained(OpId from, OpId to) const {
    auto index = to * nBitSetsPerOp + from / BitSetSize;
    auto shift = from % BitSetSize;
    return fwdEdgeSet[index][shift];
  }

  /**
   * Return true of there is no constraint a->b and no constraint b->a. In
   * other words, returns true if and only if (iff) there exists at least 1
   * schedule with a before b, and at least 1 schedule with b before a. This
   * query is performed in O(1) time.
   */
  bool unconstrained(OpId a, OpId b) const {
    return !constrained(a, b) && !constrained(b, a);
  }

  /**
   * Objects used to select sub-sets of Ops based on constraints with respect
   * to other Ops. Filters are used in get(.) with the following semantics
   *  {IsFirst::Yes, a}
   *      will be true for all b s.t. b is before a in all schedules,
   *  {IsFirst::Maybe, a}
   *      will be true for all b s.t. b is before a in at least 1 schedule,
   *      and b is after a in at least 1 schedule,
   *  {IsFirst::No, a}
   *      will be true for all b s.t. b is after a in all schedules.
   * */
  using Filter  = std::tuple<IsFirst, OpId>;
  using Filters = std::vector<Filter>;

  /**
   * Return the intersection of Filters. Example:
   * get({{IsFirst::Yes, a}, {IsFirst::Maybe, b}, {IsFirst::No, c}}) returns
   * the set of all Ops which are (1) always after a (as the first filter is
   *"a is IsFirst::Yes") (2) sometimes before a, sometimes after b and (3)
   *always before c.
   **/
  std::vector<OpId> get(const Filters &) const;

  /** The size of the set which would be returned by get(fs) */
  uint64_t n(const Filters &fs) const;

  /** Return true iff get({{x, y}}) is the same for all y in ys */
  bool same(IsFirst x, const std::vector<OpId> &ys) const;

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

  /**
   * For each Op op in subOps, what can be said about op's position in any
   * schedule relative to each of the other Ops in subOps? For example, if op
   * appears before all op2 in subOps (where op != op2) in all schedules, then
   * op has IsFirst::Yes returned from this function. As an example, suppose
   * the DAG is a->{b,c} b->{d} c->{d} d->{}. Then getExtremumStatuses({a,b})
   * is {{IsFirst::Yes, IsFinal::No}, {IsFirst::No, IsFinal::Yes}}.
   */
  std::vector<std::tuple<IsFirst, IsFinal>>
  getExtremumStatuses(const std::vector<OpId> &subOps) const;

  /**
   * Get the relative position of #opId within #subset.
   *
   *
   * Example 1:
   *  subset = {opId, foo, bar} and the underlying DAG is,
   *      opId -> foo -> bar
   * then (IsFirst::Yes, IsFinal::No) is returned.
   *
   * Example 2:
   *  subset = {opId, foo, bar} and the underlying DAG is,
   *        +-> opId
   *  bar --+
   *        +-> foo
   * then (IsFirst::No, IsFinal::Maybe) is returned. This is because there
   * are 2 possible schedules, and opId doesn't appear first in either of
   * them, but does appear last (final) in 1 of them.
   *
   *
   * Example 3:
   *  subset = {opId, foo, bar} and the underlying DAG is,
   *  opId --+
   *         +--> bar
   *  foo ---+
   *  then (IsFirst::Maybe, IsFinal::No) is returned.
   *
   *
   * In the above examples, #opId is included in #subset, but it needn't be.
   * */
  std::tuple<IsFirst, IsFinal>
  getExtremumStatus(OpId opId, const std::vector<OpId> &subset) const;

  /** Return a set of Edges which could be removed without changing the
   * Closure */
  std::vector<std::array<OpId, 2>>
  getFlattenedRedundants(const Edges &) const;
  Edges getRedundants(const Edges &) const;

  /** Amongst all schedules, what is the earliest that "id" appears ? */
  uint64_t earliest(OpId id) const;

  /** Amongst all schedules, what is the latest that "id" appears ? */
  uint64_t latest(OpId id) const;

  /** Returns true if all Ops which are unconstraned with respect to id, have
   * their earliest possible schedulings no earlier than id's */
  bool asEarlyAsAllUnconstrained(OpId id) const;

  bool operator==(const TransitiveClosure &x) const {
    return fwdEdgeSet == x.fwdEdgeSet && bwdEdgeSet == x.bwdEdgeSet;
  }

  bool operator!=(const TransitiveClosure &x) const { return !operator==(x); }

  /** Only advanced optimizations should require direct bit-access */
  const std::vector<BitSet> &getFwdEdgeSet() const { return fwdEdgeSet; }
  const std::vector<BitSet> &getBwdEdgeSet() const { return bwdEdgeSet; }

  uint64_t getNBitSetsPerOp() const { return nBitSetsPerOp; }

private:
  uint64_t nOps;
  uint64_t nBitSetsPerOp;
  uint64_t nBitSets;

  std::vector<BitSet> fwdEdgeSet;
  std::vector<BitSet> bwdEdgeSet;

  void insertConstraint(OpId from, OpId to, std::vector<BitSet> &edgeSet);
  std::vector<BitSet> getBits(const Filters &) const;
};

std::ostream &operator<<(std::ostream &,
                         schedule::transitiveclosure::IsFirst);
std::ostream &operator<<(std::ostream &,
                         schedule::transitiveclosure::IsFinal);

} // namespace transitiveclosure
} // namespace schedule

} // namespace poprithms

#endif
