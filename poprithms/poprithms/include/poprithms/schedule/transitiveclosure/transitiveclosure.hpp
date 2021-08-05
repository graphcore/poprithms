// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_TRANSITIVECLOSURE_TRANSITIVECLOSURE_HPP
#define POPRITHMS_SCHEDULE_TRANSITIVECLOSURE_TRANSITIVECLOSURE_HPP

#include <array>
#include <bitset>
#include <tuple>
#include <vector>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

// The number of bits stored per std::bitset, see bitset_performance_0.cpp for
// some experiments to choose this value. std::bitset is a good data-type to
// use to store the transitive closure bits, as it is compact, and it has a
// fast count() method (probably compiled to x86's popcnt instruction).
static constexpr uint64_t BitSetSize = 512;

using BitSet  = std::bitset<BitSetSize>;
using BitSets = std::vector<BitSet>;

using OpId  = uint64_t;
using OpIds = std::vector<OpId>;
using Edges = std::vector<OpIds>;

enum class IsFirst { No = 0, Maybe, Yes };
enum class IsFinal { No = 0, Maybe, Yes };

std::ostream &operator<<(std::ostream &,
                         const std::tuple<IsFirst, IsFinal> &);

/**
 * A class for compactly storing all indirect topological constraints between
 * Nodes (Ops) in a DAG.
 *
 * Example: suppose constraints between a,b and c are
 *   a -> b
 *   b -> c.
 * Then the transitive closure will contain a->c as this is an indirect
 * constraint. For more info, see:
 *
 * https://en.wikipedia.org/wiki/Transitive_closure
 */
class TransitiveClosure {

public:
  /**
   * Construct a transitive closure from a set for forward edges of a DAG.
   */
  explicit TransitiveClosure(const Edges &forwardEdges);

  /**
   * Insert additional DAG edges. Note that it is much faster to call the
   * constructor with the full set of edges, than to sequentially call update
   * on each of the edges individually.
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
   * Return true if there is no constraint a->b and no constraint b->a. In
   * other words, returns true if and only if (iff) there exists at least 1
   * schedule with a before b, and at least 1 schedule with b before a. This
   * query is performed in O(1) time.
   */
  bool unconstrainedInBothDirections(OpId a, OpId b) const {
    return !constrained(a, b) && !constrained(b, a);
  }

  /**
   * Objects used to select sub-sets of Ops based on constraints with respect
   * to other Ops. These filters are used to express the following semantics:
   *
   *  {IsFirst::Yes, a}
   *      will be true for all b s.t. b is before a in all schedules,
   *
   *  {IsFirst::Maybe, a}
   *      will be true for all b s.t. b is before a in at least 1 schedule,
   *      and b is after a in at least 1 schedule,
   *
   *  {IsFirst::No, a}
   *      will be true for all b s.t. b is after a in all schedules.
   * */
  using Filter  = std::tuple<IsFirst, OpId>;
  using Filters = std::vector<Filter>;

  /**
   * Return the intersection of Filters #filters.
   * Example:
   * <code>
   *    opIntersection
   *         ({{IsFirst::Yes, a}, {IsFirst::Maybe, b}, {IsFirst::No, c}});
   * </code>
   * returns the set of all Ops which are
   *    (1) always after a (as the first filter is "a is IsFirst::Yes"), AND
   *    (2) sometimes before a, sometimes after b, AND
   *    (3) always before c.
   **/
  OpIds opIntersection(const Filters &filters) const;

  /**
   * The number of Ops which are in the intersection of #filters.
   *
   * \sa opIntersection.
   * */
  uint64_t nIntersection(const Filters &filters) const;

  /**
   * The union of Filters #filters. That is, the set of all Ops which are true
   * for at least one of the filters.
   * */
  OpIds opUnion(const Filters &filters) const;

  /**
   * The number of Ops which satisfy at least one of the Filters in #filters.
   * */
  uint64_t nUnion(const Filters &filters) const;

  /**
   * \return All Ops which satisfy Filter #f.
   **/
  OpIds get(const Filter &f) const { return opIntersection({f}); }

  /**
   * \return The number of Ops which satisfy Filter #f.
   * */
  uint64_t n(const Filter &f) const { return nIntersection({f}); }

  /**
   * Return true iff get({isFirst, opId}) is the same for all opId in ids. In
   * other words, return true when the set of Ops which always precede the Ops
   * in #ids is the same.
   */
  bool same(IsFirst isFirst, const OpIds &ids) const;

  /**
   * Return all Ops which can be scheduled either before #id, or after #id.
   * */
  OpIds getUnconstrained(OpId id) const { return get({IsFirst::Maybe, id}); }

  /**
   * Return all Ops which are always scheduled after #id.
   * */
  OpIds getPost(OpId id) const { return get({IsFirst::No, id}); }

  /**
   * Return all Ops which are unconstrained with respect to #a, and always
   * after #b.
   * */
  OpIds getUnconstrainedPost(OpId a, OpId b) const {
    return opIntersection({{IsFirst::Maybe, a}, {IsFirst::No, b}});
  }

  /**
   * Return true if the set of Ops which are unconstrained with respect to #a
   * is the same as the set of Ops which are unconstrained with respect to #b.
   * */
  bool sameUnconstrained(OpId a, OpId b) const {
    return same(IsFirst::Maybe, {a, b});
  }

  /**
   * Return the number of Ops which are always after #a, and always after #b.
   * */
  uint64_t nPostPost(OpId a, OpId b) const {
    return nIntersection({{IsFirst::No, a}, {IsFirst::No, b}});
  }

  uint64_t nOps_u64() const { return nOps; }
  int64_t nOps_i64() const { return static_cast<int64_t>(nOps); }

  /**
   * For each Op op in #subOps, what can be said about its position in any
   * schedule relative to each of the other Ops in subOps?
   *
   * For example, if op1 appears before all op2 in subOps (where op1 != op2)
   * in all schedules, then op1 has IsFirst::Yes returned from this function.
   *
   * As another example, suppose the DAG is a->{b,c} b->{d} c->{d} d->{}:
   *
   *    a --+--- b --->-+--- d
   *        |           |
   *        +--- c --->-+
   *
   * Then,
   * getExtremumStatuses({a,b}) returns
   *   {{IsFirst::Yes, IsFinal::No},
   *    {IsFirst::No, IsFinal::Yes}},
   *
   * because a is always before b. getExtremumStatuses({a,b,c}) returns
   *   {{IsFirst::Yes, IsFinal::No},
   *    {IsFirst::No, IsFinal::Maybe},
   *    {IsFirst::No, IsFinal::Maybe}}.
   */
  std::vector<std::tuple<IsFirst, IsFinal>>
  getExtremumStatuses(const OpIds &subOps) const;

  /**
   * Get the relative position of #opId within #subset.
   *
   * Example 1:
   *  subset = {opId, foo, bar} and the underlying DAG is,
   *      opId -> foo -> bar
   * then (IsFirst::Yes, IsFinal::No) is returned, because #opId is always
   * first and never last.
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
  std::tuple<IsFirst, IsFinal> getExtremumStatus(OpId opId,
                                                 const OpIds &subset) const;

  /** Return a set of Edges which could be removed without changing the
   * closure of the DAG given by #edges. */
  std::vector<std::array<OpId, 2>>
  getFlattenedRedundants(const Edges &edges) const;
  Edges getRedundants(const Edges &) const;

  /** Amongst all schedules, what is the earliest that #id appears ? */
  uint64_t earliest(OpId id) const { return n({IsFirst::Yes, id}); }

  /** Amongst all schedules, what is the latest that #id appears ? */
  uint64_t latest(OpId id) const { return nOps - n({IsFirst::No, id}) - 1; }

  /**
   * Returns true if all Ops which are unconstrained with
   * respect to id, have their earliest possible schedulings no earlier than
   * id's
   * */
  bool asEarlyAsAllUnconstrained(OpId id) const;

  bool operator==(const TransitiveClosure &x) const;

  bool operator!=(const TransitiveClosure &x) const { return !operator==(x); }

  /**
   * For each Op #id, this class stores bitsets representing all of the
   * forward and backward constraints with all other Ops. These bitsets come
   * in chunks of bits of size BitSetSize (see comment at start of class).
   *
   * This method checks for constraints between Op #id and all Ops with ids in
   * [bitSetIndex*BitSetSize, (bitSetIndex + 1)*bitSetSize).
   *
   * This method is used for advanced, performance critical use cases.
   * */
  bool unconstrainedWithAtLeastOne(OpId, uint64_t bitSetIndex) const;

  // An interval [low, high).
  struct DurationBound {
    uint64_t low;
    uint64_t high;
    bool operator==(const DurationBound &r) const {
      return low == r.low && high == r.high;
    }
    bool operator!=(const DurationBound &r) const { return !operator==(r); }
  };

  /**
   * Consider these 2 extremes over all possible schedules,
   *
   * MIN = min_{all schedules}
   *          (max-schedule-index(opIds) - min-schedule-index(opIds)), and
   *
   * MAX = max_{all schedules}
   *          (max-schedule-index(opIds) - min-schedule-index(opIds)).
   *
   * This method returns bounds on MAX and MIN. In particular, it is
   * guaranteed that the returned DurationBound has
   *    1) low <= MIN, and
   *    2) high > MAX,
   *
   * If low = high+1, it is guaranteed that
   *       max-schedule-index(opIds) - min-schedule-index(opIds) = low,
   *
   * for all schedules.
   *
   * */
  DurationBound getDurationBound(const OpIds &opIds) const;

  /**
   * the total size of all bitmaps used by this object
   * */
  uint64_t nBits() const {
    return (fwdEdgeSet.size() + bwdEdgeSet.size()) * BitSetSize;
  }

  uint64_t getNBitSets(OpId) const;

  static uint64_t getNBitSetsPerOp(uint64_t nOps) {
    return nOps / BitSetSize + (nOps % BitSetSize != 0);
  }

private:
  uint64_t nOps;
  uint64_t nBitSetsPerOp;

  BitSets fwdEdgeSet;
  BitSets bwdEdgeSet;

  // all true bits, except for the tail (the final nOps % BitSetSize bits).
  static BitSets getAllTrue(uint64_t nOps);

  // all false bits.
  static BitSets getAllFalse(uint64_t nOps);

  // the number of true bits in #bs.
  static uint64_t n(const BitSets &bs);

  // Combiner is either an object which merges with intersection or union.
  template <typename Combiner>
  BitSets bitSetCombine(Filters filters, Combiner &&) const;

  // Combiner is either an object which merges with intersection or union.
  template <typename Combiner>
  BitSets bitSetCombine(const std::vector<BitSets> &, Combiner &&) const;

  void insertConstraint(OpId from, OpId to, BitSets &edgeSet);

  BitSets bitSetIntersection(const std::vector<BitSets> &) const;
  BitSets bitSetIntersection(const Filters &) const;

  BitSets bitSetUnion(const std::vector<BitSets> &) const;
  BitSets bitSetUnion(const Filters &) const;

  BitSet getIsFirstMaybe(OpId, uint64_t bitsetIndex) const;

  class Intersecter;
  class Unioner;

public:
  static OpIds opIds(const BitSets &);
};

std::ostream &operator<<(std::ostream &,
                         schedule::transitiveclosure::IsFirst);

std::ostream &operator<<(std::ostream &,
                         schedule::transitiveclosure::IsFinal);

std::ostream &operator<<(std::ostream &,
                         const TransitiveClosure::DurationBound &);

std::ostream &operator<<(std::ostream &, const BitSets &);

std::ostream &operator<<(std::ostream &, const TransitiveClosure &);

} // namespace transitiveclosure
} // namespace schedule

} // namespace poprithms

#endif
