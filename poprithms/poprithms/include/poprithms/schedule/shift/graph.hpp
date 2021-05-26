// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_GRAPH
#define POPRITHMS_SCHEDULE_SHIFT_GRAPH

#include <array>
#include <map>
#include <tuple>
#include <vector>

#include <poprithms/schedule/shift/alloc.hpp>
#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/schedule/shift/op.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * A minimal graph representation for tensor liveness-based scheduling.
 *
 * A Graph consists of:
 *
 * 1) operations (Ops).
 *
 * 2) topological constraints between Ops, which constrain the schedule. A
 *    constraint (a,b) means that only schedules where a appears before b are
 *    valid.
 *
 * 3) links between Ops. These are contiguous topological constraints. A link
 *    (a,b) means that only schedules where a appears directly before b, are
 *    valid.
 *
 * 4) allocations (Allocs) which are required to be live when certain Ops are
 *    scheduled.
 *
 * With these basic components, more elaborate components can be constructed.
 * For example
 *
 * 1) bin constraints, where one set of Ops must appear before another set.
 *    This can be done by creating a single "bottleneck" Op between the 2
 *    sets, which means only O(N) constraints are needed, connecting each
 *    element in the sets to the bottleneck, instead of O(N^2).
 *
 * 2) Op attractions, which are like "soft" links. This is done by assigning
 *    an Alloc to the 2 Ops which are attracted.
 *
 * There are helper functions in this class for 1 and 2, which will be
 * discussed in more detail later. Note that some of these helper functions
 * insert additional ops into the graph.
 * */

class Graph {
public:
  Graph()  = default;
  ~Graph() = default;

  Graph(Graph &&)      = default;
  Graph(const Graph &) = default;

  Graph &operator=(Graph &&) = default;
  Graph &operator=(const Graph &) = default;

  template <typename T> Graph(const std::vector<std::vector<T>> &fwd) {
    for (uint64_t i = 0; i < fwd.size(); ++i) {
      insertOp("");
    }
    for (uint64_t i = 0; i < fwd.size(); ++i) {
      for (auto out : fwd[i]) {
        insertConstraint(i, out);
      }
    }
  }

  /**
   * Comparison of Graphs is not a true graph isomorphism, the order in which
   * Ops and constraints are inserted matters.
   * */
  bool operator==(const Graph &rhs) const;
  bool operator!=(const Graph &rhs) const;
  /**
   * Optionally ignore the names (debug strings) of Ops in the comparison.
   * */
  bool equalTo(const Graph &, bool includeNames) const;

  bool operator<(const Graph &rhs) const;
  bool operator<=(const Graph &rhs) const;
  bool operator>(const Graph &rhs) const;
  bool operator>=(const Graph &rhs) const;
  bool lessThan(const Graph &, bool includeNames) const;

  /**
   * Create an Alloc in this Graph.
   *
   * \param w The "size" of the Allocation.
   * \return An AllocAddress, which uniquely identifies the Alloc created.
   * */
  AllocAddress insertAlloc(AllocWeight w);

  AllocAddress insertAlloc(double w) { return insertAlloc({w, 0}); }

  /**
   * Create an Op in this Graph.
   *
   * \param dbString A string used in logging, associated to the Op created.
   * \return An OpAddress, which uniquely identifies this Op created.
   * */
  OpAddress insertOp(const std::string &dbString);

  /**
   * Create multiple Ops in this Graph.
   *
   * \param dbStrings Strings used in logging, one to associate with each Op.
   * \return OpAddresses which uniquely identify the Ops created.
   * */
  std::vector<OpAddress> insertOps(const std::vector<std::string> &dbStrings);

  /**
   * Register that #aa must be live when #oa is scheduled.
   * */
  void insertOpAlloc(OpAddress oa, AllocAddress aa);

  /**
   * Register that #aa must be live when each Op in #oas are scheduled.
   * */
  void insertOpAlloc(const std::vector<OpAddress> &oas, AllocAddress aa);

  /**
   * Register that #before must execute before #after.
   * */
  void insertConstraint(OpAddress before, OpAddress after);

  /**
   * Register multiple constraints.
   * */
  using BeforeAndAfter = std::array<OpAddress, 2>;
  void insertConstraints(const std::vector<BeforeAndAfter> &constraints);

  /**
   * Register that #before must execute before #after, and that no other Ops
   * can be scheduled between #before and #after.
   * */
  void insertLink(OpAddress before, OpAddress after);

  /**
   * Insert an Op, and simultaneously register topological constraints and
   * liveness conditions.
   *
   * \param befores Ops which must appear before the Op being created.
   * \param allocs Allocs which must be live when the Op being created is
   *               scheduled.
   *
   * \param dbString A logging string to associate to the Op being created.
   * */
  template <typename A, typename B>
  OpAddress insertOp(A &&befores, B &&allocs, const std::string &dbString) {
    auto opId = insertOp(dbString);
    for (auto &&x : befores) {
      insertConstraint(x, opId);
    }
    for (auto &&x : allocs) {
      insertOpAlloc(opId, x);
    }
    return opId;
  }

  /**
   * Insert an Op, and simultaneously register topological constraints and
   * liveness conditions.
   *
   * \param befores Ops which must appear before the Op being created.
   * \param allocs Allocs which are live when the Op being created is
   *               scheduled.
   * \param dbString A logging string to associate to the Op being created.
   * */
  OpAddress insertOp(std::initializer_list<OpAddress> befores,
                     std::initializer_list<AllocAddress> as,
                     const std::string &dbString) {
    return insertOp(std::vector<OpAddress>(befores),
                    std::vector<AllocAddress>(as),
                    dbString);
  }

  /**
   * Generate a new Graph by merging groups of Ops in this Graph into single
   * Ops. The returned tuple consists of (1) the reduced Graph, containing
   * merged Ops and (2) a mapping from the Ops in the reduced (child) Graph to
   * Ops in this (the parent) Graph.
   * */
  using ParentGraphOps = std::vector<std::vector<OpAddress>>;
  using OpMerged       = std::tuple<Graph, ParentGraphOps>;
  OpMerged getMerged(const std::vector<std::vector<OpAddress>> &chains) const;

  /**
   * Merge all chains formed of Ops with Links. Recall that linked Ops are
   * guarenteed to be scheduled contiguously.
   * */
  OpMerged getLinkMerged() const;

  /**
   * Merge all chains formed of tightly paired Ops. Recall that two Ops are
   * said to be tightly paired if one is the unique output of the other, which
   * in turn is the unique input of the first.
   * */
  OpMerged getTightMerged() const;

  static Graph fromSerializationString(const std::string &);
  void appendSerialization(std::ostream &) const;
  std::string getSerializationString() const;

  void append(std::ostream &ost) const;

  const std::vector<Op> &getOps() const { return allOps; }

  const Op &getOp(OpAddress address) const { return allOps[address]; }

  /**
   * \return The total number of Ops in this Graph.
   * */
  uint64_t nOps() const { return allOps.size(); }

  int nOps_i32() const { return static_cast<int>(nOps()); }

  /**
   * \return The total number of constraints in this Graph.
   * */
  uint64_t nConstraints() const;

  const std::vector<Alloc> &getAllocs() const { return allAllocs; }

  const Alloc &getAlloc(AllocAddress a) const { return allAllocs[a]; }

  uint64_t nAllocs() const { return getAllocs().size(); }

  /**
   * \return All Ops which do not have any input dependencies. That is, Ops
   *         which appear first in at least 1 valid schedule.
   * */
  std::vector<OpAddress> getInputOps() const;

  /**
   * Convenience function for inserting constraints between groups of Ops.
   *
   * \param bins Ops in different elements of bins must be scheduled in
   *             increasing bin index. For example, if a is in bins[0] and b
   *             is in bins[2], then a must appear before b in the schedule.
   *
   * \param opPrefix The implementation of this method inserts a bottleneck Op
   *                 between the groups, as this is more efficient than
   *                 inserting all individual constraints between Ops. This
   *                 string will be associated to the bottleneck Op(s).
   * */
  void insertBinConstraints(const std::vector<std::vector<OpAddress>> &bins,
                            const std::string &opPrefix);

  /**
   * \param pairs Pair (a,b) in pairs should appear close to each other in
   *              the schedule, where the "force of attraction" is determined
   *              by w.
   *
   * \param w the importance associated to having Ops of a pair close
   *          each other in the schedule. In particular, for each pair, an
   *          Alloc is created of size w, and associated to the 2 Ops in the
   *          pair.
   *  */
  void insertAttractions(const std::vector<std::array<OpAddress, 2>> &pairs,
                         AllocWeight w);

  /**
   * A pair of Ops (a,b) is defined to be a "tight pair" if
   *   1) b is the only output of a,
   *   2) a is the only input of b.
   *
   * Let C(a) be the set of all Ops c s.t. there is no implicit constraint
   * between a and c. It is easy to see that (a,b) is tight implies C(a) =
   * C(b), but C(a) = C(b) does not imply (a,b) is tight.
   *  */
  std::vector<std::array<OpAddress, 2>> getTightPairs() const;

  /**
   * Starting from #a and proceeding through Op outputs, find a chain of
   * tightly paired Ops. The returned vector may be the singleton, {a}, if it
   * is not tightly coupled to an output.
   * */
  std::vector<OpAddress> tightChainFrom(OpAddress a) const;

  /**
   * All constraints which are in this Graph, but not in "rhs".
   * */
  std::vector<std::vector<OpAddress>>
  constraintDiff(const std::vector<std::vector<OpAddress>> &rhs) const;

  std::vector<std::vector<OpAddress>> constraintDiff(const Graph &rhs) const {
    return constraintDiff(rhs.getForwardEdges());
  }

  /**
   * Return all Ops which have the same ins as #a. #a is an element of the
   * returned vector.
   * */
  std::vector<OpAddress> getIdenticalIns(OpAddress a) const;

  std::vector<std::vector<OpAddress>> getForwardEdges() const;

  // Combine all linked Ops form a sets of isolated chains
  std::vector<std::vector<OpAddress>> getLinkChains() const;

  // Combine all tight Op pairs to form sets of isolated chains
  std::vector<std::vector<OpAddress>> getTightChains() const;

  std::vector<OpAddress> getOpsWithFwdLinks() const;

  /**
   * For each Op #a in #opAddresses, insert an Op #proxy, which is constrained
   * to be scheduled very early, and 1 Alloc, which must be live for #proxy
   * and Op #a; this attracts #a towards the beginning of the schedule; tue
   * Allocs' AllocWeights, which determine the force of attraction of #a to
   * the beginning of the schedule, determined by #relativeLexico and
   * #stepSize.
   * */
  template <typename T>
  void insertStartAttractors(const std::vector<OpAddress> &opAddresses,
                             const std::vector<T> &priorities,
                             int relativeLexico,
                             double stepSize = 1.0) {

    // For each Op "a" in opAddresses, the size of the attracting Alloc is
    // determined by the corresponding priority in "priorities"
    insertStartAttractorsAssert0(opAddresses.size(), priorities.size());

    // All Ops which have no dependencies and can legally be executed first
    auto inputs = getInputOps();

    // sort and unique-ify the priorities
    std::vector<T> unipris(priorities);
    std::sort(unipris.begin(), unipris.end());
    auto last = std::unique(unipris.begin(), unipris.end());
    unipris.erase(last, unipris.cend());

    // If all the priorities are the same, then return - giving all Ops the
    // same level attraction to the start is equivalent to giving them all no
    // attraction to the start.
    if (unipris.size() <= 1) {
      return;
    }

    // Give each unique T a corresponding AllocWeight:
    std::map<T, AllocWeight> ws;
    for (uint64_t i = 0; i < unipris.size(); ++i) {
      ws.insert(
          {unipris[i],
           AllocWeight(stepSize * static_cast<double>(i), relativeLexico)});
    }

    std::vector<OpAddress> attractors;

    for (uint64_t i = 0UL; i < opAddresses.size(); ++i) {
      auto opAddress = opAddresses[i];
      auto pri       = priorities[i];
      auto w         = ws.at(pri);

      if (w != AllocWeight(0)) {
        auto allocAddress = insertAlloc(w);

        std::string attractorStr{"priorityAttractor_" +
                                 getOp(opAddress).getDebugString() + "_" +
                                 toString(w)};

        auto attractor = insertOp({}, {allocAddress}, attractorStr);

        insertOpAlloc(opAddress, allocAddress);
        attractors.push_back(attractor);
      }
    }

    // force attractors to be in a fixed order at the start of the schedule
    for (auto x = std::next(attractors.cbegin()); x != attractors.cend();
         std::advance(x, 1)) {
      insertConstraint(*std::prev(x), *x);
    }
    for (auto x : inputs) {
      insertConstraint(*attractors.crbegin(), x);
    }
  }

  // extract all forward edges from the Ops.
  std::vector<std::vector<uint64_t>> getFwdEdges_u64() const;

  void removeConstraint(OpAddress before, OpAddress after);

private:
  std::vector<Op> allOps;
  std::vector<Alloc> allAllocs;

  void insertStartAttractorsAssert0(uint64_t, uint64_t) const;
};

std::ostream &operator<<(std::ostream &ost, const Graph &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
