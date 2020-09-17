// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_ANNEAL_GRAPH
#define POPRITHMS_SCHEDULE_ANNEAL_GRAPH

#include <array>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include <poprithms/schedule/anneal/alloc.hpp>
#include <poprithms/schedule/anneal/allocweight.hpp>
#include <poprithms/schedule/anneal/annealusings.hpp>
#include <poprithms/schedule/anneal/op.hpp>
#include <poprithms/schedule/anneal/schedulechange.hpp>
#include <poprithms/schedule/anneal/shiftandcost.hpp>
#include <poprithms/schedule/anneal/trackentry.hpp>
#include <poprithms/schedule/anneal/transitiveclosureoptimizations.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

/// Implementations of the sum-liveness minimizing algorithm. They differ only
/// in time to solution, the final schedule obtained using these is identical.
/// RIPPLE is much faster.
enum class MinSumLivenessAlgo {
  SIMPLE, ///< A simple implementation, for debugging and understanding
  RIPPLE  ///< An optimized implementation which eliminates certain redundant
          ///< computations. It's name derives from the way it re-uses results
          ///< using dynamic programming across consecutive schedule indices.
};

/// The core sum-liveness minimizing algorithm is preceded by a single run of
/// Kahn's algorithm to obtain an initial, valid schedule. Kahn's algorithm
/// requires a "tie-breaker" when more than 1 Op is schedulable. Three
/// tie-breakers are implemented:
enum class KahnTieBreaker {
  RANDOM = 0, ///< Choose an Op at random
  GREEDY, ///< Choose the Op which results in the largest immediate liveness
          ///< reduction.
  FIFO,   ///< Choose the Op which became available most recently (this should
          ///< be called FILO).
  N ///< Not a tie-breaker: this is the number of tie-breakers listed above.
};

static constexpr auto NKahnTieBreakers =
    static_cast<uint64_t>(KahnTieBreaker::N);
std::ostream &operator<<(std::ostream &, KahnTieBreaker);
KahnTieBreaker kahnTieBreaker(const std::string &);

/**
 * A minimal Graph representation for Tensor liveness-based scheduling. The
 * Graph consists of Ops, the topological constraints between them, and the
 * Allocs which are required to be live when certain Ops execute.
 *
 * The core algorithm implemented for this Graph class attempts to minimize
 * the sum of the livenesses of the Allocs, where an Alloc is live from the
 * first to last of its Ops' schedule indices.
 *
 * For example, if an Alloc 'a' has Ops {'b','c','d'} which require it to be
 * live, and the schedule indices of 'b','c', and 'd' are 5,8 and 11
 * respectively, then 'a' is live for a duration of 11 - 5 + 1 = 7. Further
 * information is available in the notes directory of poprithms.
 *
 * A Graph is grown incrementally with functions for inserting Ops, Allocs and
 * constraints between Ops.
 *
 * Some helper functions for growing the graph, e.g. for inserting bin
 * constraints, will also insert extra "internal" Ops to the graph. These
 * functions should state so in their documentation, and allow the user to
 * give such ops a debug prefix to their name.
 *
 * It is possible to both view the full internal schedule, containing all the
 * internal Ops, and to get the sub-schedule for only a specified subset of
 * Ops.
 * */

class Graph {
public:
  /**
   * Create an Alloc in this Graph.
   *
   * \param w The "size" of the Allocation.
   *
   * \return An AllocAddress, which uniquely identifies the Alloc created.
   * */
  AllocAddress insertAlloc(AllocWeight w);

  AllocAddress insertAlloc(double w) { return insertAlloc({w, 0}); }

  /**
   * Create an Op in this Graph.
   *
   * \param dbString A string used in logging, associated to the Op created.
   *
   * \return An OpAddress, which uniquely identifies this Op created.
   * */
  OpAddress insertOp(const std::string &dbString);

  /**
   * Create multiple Ops in this Graph.
   *
   * \param dbStrings Strings used in logging, one to associate with each Op.
   *
   * \return OpAddresses which uniquely identify the Ops created.
   * */
  std::vector<OpAddress> insertOps(const std::vector<std::string> &dbStrings);

  /** Register that "aa" must be live when "oa" is scheduled */
  void insertOpAlloc(OpAddress oa, AllocAddress aa);

  /** Register that "aa" must be live when each Op in "oas" are scheduled */
  void insertOpAlloc(const std::vector<OpAddress> &oas, AllocAddress aa);

  /** Register that "before" must execute before "after" */
  void insertConstraint(OpAddress before, OpAddress after);

  /** Register multiple "before" -> "after" constraints */
  using BeforeAndAfter = std::array<OpAddress, 2>;
  void insertConstraints(const std::vector<BeforeAndAfter> &css);

  /** Register that "before" must execute before "after", and that no other
   * Ops can be scheduled between "before" and "after". */
  void insertLink(OpAddress before, OpAddress after);

  /**
   * Insert an Op, and simultaneously register topological constraints and
   * liveness conditions.
   *
   * \param befores Ops which must appear before the Op being created
   *
   * \param allocs Allocs which must be live when the Op being created is
   *               scheduled
   *
   * \param dbString A logging string to associate to the Op being created
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
   * Create an Op from a set of topological constraints, Alloc conditions,
   * and a debug string.
   *
   * \param befores Ops which must appear before the Op being created.
   *
   * \param allocs Allocs which are live when the Op being created is
   *               scheduled.
   *
   * \param dbString A logging string to associate to the Op being created.
   * */
  OpAddress insertOp(std::initializer_list<OpAddress> befores,
                     std::initializer_list<AllocAddress> allocs,
                     const std::string &dbString) {
    return insertOp(std::vector<OpAddress>(befores),
                    std::vector<AllocAddress>(allocs),
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
   * Merges all chains formed of Ops with Links. Recall that linked Ops are
   * guarenteed to be scheduled contiguously.
   * */
  OpMerged getLinkMerged() const;

  // Merges all chains formed of tightly paired Ops
  // Recall : two Ops are said to be tightly paired if one is the unique
  // output of the other, which in turn is the unique input of the first.
  OpMerged getTightMerged() const;

  static Graph fromSerializationString(const std::string &);
  void appendSerialization(std::ostream &) const;
  std::string getSerializationString() const;

  void append(std::ostream &ost) const;

  const std::vector<Op> &getOps() const { return allOps; }
  const Op &getOp(OpAddress address) const { return allOps[address]; }
  uint64_t nOps() const { return allOps.size(); }
  int nOps_i32() const { return static_cast<int>(nOps()); }

  /** \return The total number of constraints */
  uint64_t nConstraints() const;

  const std::vector<Alloc> &getAllocs() const { return allAllocs; }
  const Alloc &getAlloc(AllocAddress address) const {
    return allAllocs[address];
  }
  uint64_t nAllocs() const { return getAllocs().size(); }
  std::string getLivenessString() const;

  /** Initialize the Graph. This method should be called once, after the all
   * Op and Alloc insertions and associations are complete
   *
   * \param ktb The Method by which to choose an Op from a set which are ready
   *            to be scheduled
   *
   * \param kahnSeed For the RANDOM tie-breaker, the initial seed.
   *
   * \param tco The set of Optimizations to apply to the Graph, to accelerate
   *            the min-sum-liveness algorithm. These optimizations insert
   *            constraints and links between Ops which all sum-liveness
   *            minimizing schedules satisfy.
   * */
  void initialize(KahnTieBreaker ktb = KahnTieBreaker::GREEDY,
                  uint32_t kahnSeed  = defaultKahnSeed(),
                  TransitiveClosureOptimizations tco =
                      TransitiveClosureOptimizations::allOff());

  void initialize(const std::map<std::string, std::string> &);

  /** A method to be called once after growing
   */
  void finalize();

  /** \return false iff there exists a cycle or incompatible Links */
  bool isSchedulable() const;

  /** verify that all graph connections are valid, if not throw error */
  void assertCorrectness() const;

  static bool defaultDebug() { return false; }
  static uint32_t defaultMinSumLivenessSeed() { return 1; }
  static double defaultPStayPut() { return 10.0; }
  static double defaultPHigherFallRate() { return 2.0; }
  static double defaultPClimb() { return 1.0; }
  static bool defaultFilterSusceptible() { return true; }
  static double defaultTimeLimitSeconds() { return 1e9; }
  static int64_t defaultSwapLimitCount() { return static_cast<int64_t>(1e9); }

  static KahnTieBreaker defaultKahnTieBreaker() {
    return KahnTieBreaker::GREEDY;
  }
  static uint32_t defaultKahnSeed() { return 1; }
  static TransitiveClosureOptimizations
  defaultTransitiveClosureOptimizations() {
    // TODO(T19732) change to allOn(). Make sure all buildbots are happy with
    // this before landing.
    return TransitiveClosureOptimizations::allOff();
  }

  /**
   * All Ops which do not have any input dependencies. That is, Ops which
   * appear first in at least 1 valid schedule */
  std::vector<OpAddress> getInputOps() const;

  /**
   * The core optimization algorithm of this class. Some preliminaries:
   *
   * Definition of sum-liveness: the sum over all schedule indices of the
   * AllocWeights of the Allocs which are live.
   *
   * Definition of a round: One iteration through all Ops to search for, and
   * possibly apply, sum-liveness reducing improvements.
   *
   * After each round with at least 1 improvement, the algorithm runs again
   * with the same nToShift (see notes directory for definition of nToShift).
   *
   * \param algo Implementation to use.
   *
   * \param debug Compares algo (above) to SIMPLE to confirm agreement, and
   *              checks state of graph edges at each iteration. debug=true
   *              makes execution slow.
   *
   * \param seed  This algorithm randomly shuffles Op indices in each round,
   *              this random seed controls the shuffle permutation.
   *
   * \param filterSusceptible If there were shifts at the previous nToShift,
   *                          only consider shifting ranges that contain at
   *                          least one Op constrained to an Op that was
   *                          shifted in that previous round.
   *   */
  void
  minSumLivenessAnneal(MinSumLivenessAlgo algo = MinSumLivenessAlgo::RIPPLE,
                       bool debug              = defaultDebug(),
                       uint32_t seed           = defaultMinSumLivenessSeed(),
                       bool filterSusceptible  = defaultFilterSusceptible(),
                       double timeLimitSeconds = defaultTimeLimitSeconds(),
                       int64_t swapLimitCount  = defaultSwapLimitCount());

  void minSumLivenessAnneal(const std::map<std::string, std::string> &);

  AllocWeight getMaxLiveness() const;

  AllocWeight getSumLiveness() const;

  AllocWeight scheduleToLiveness(ScheduleIndex i) const {
    return schToLiveness[static_cast<uint64_t>(i)];
  }
  OpAddress scheduleToOp(ScheduleIndex i) const {
    return schToOp[static_cast<uint64_t>(i)];
  }
  ScheduleIndex opToSchedule(OpAddress a) const { return opToSch[a]; }

  // sorted schedule indices at which alloc is used
  const std::vector<ScheduleIndex> &allocToSchedule(AllocAddress a) const {
    return allocToSch[a];
  }
  ScheduleIndex allocToFirstSchedule(AllocAddress a) const {
    return allocToSch[a][0];
  }
  ScheduleIndex allocToFinalSchedule(AllocAddress a) const {
    return allocToSch[a].back();
  }

  // the allocs required by the op at a schedule index
  const std::vector<AllocAddress> &scheduleToAllocs(ScheduleIndex i) const {
    return schToAllocs[static_cast<uint64_t>(i)];
  }

  // schedule indices of an ops inputs, sorted
  const std::vector<ScheduleIndex> &opToInSchedule(OpAddress a) const {
    return opToInSch[a];
  }

  // schedule indices of an ops output, sorted
  const std::vector<ScheduleIndex> &opToOutSchedule(OpAddress a) const {
    return opToOutSch[a];
  }

  int getNCanFwd(ScheduleIndex i) const {
    return nCanFwd[static_cast<uint64_t>(i)];
  }
  int getNCanBwd(ScheduleIndex i) const {
    return nCanBwd[static_cast<uint64_t>(i)];
  }

  /**
   * View the internal schedule that this class keeps track of. This contains
   * all ops, included those added internally to implement bin constraints,
   * etc.
   *
   * If you know there were no internal ops added, you can thus use this
   * method to get a view of the schedule in constant time.
   *
   * \return Vector such that position i is the OpAddress of the i^th op in
   * the internal schedule.
   */
  const std::vector<OpAddress> &viewInternalScheduleToOp() const {
    return schToOp;
  }

  /**
   * Get the schedule, containing only the ops with the given OpAddresses.
   * This method is O(nOps).
   *
   * \param oas The OpAddresses to include in the schedule.
   *
   * \return Vector such that position i is the OpAddress of the i^th op in
   * the schedule.
   */
  std::vector<OpAddress>
  getSubSchedule(const std::vector<OpAddress> &oas) const;

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
   *  \param pairs Pair (a,b) in pairs should appear close to each other in
   *               the schedule, where the "force of attraction" is determined
   *               by w.
   *
   *  \param w the importance associated to having Ops of a pair close
   *           each other in the schedule. In particular, for each pair, an
   *           Alloc is created of size w, and associated to the 2 Ops in the
   *           pair.
   *  */
  void insertAttractions(const std::vector<std::array<OpAddress, 2>> &pairs,
                         AllocWeight w);

  bool operator==(const Graph &rhs) const {
    return allOps == rhs.allOps && allAllocs == rhs.allAllocs;
  }
  bool operator!=(const Graph &rhs) const { return !operator==(rhs); }

  // A pair of Ops (a,b) is defined to be a "tight pair" if
  // 1) b is the only output of a,
  // 2) a is the only input of b.
  // Let C(a) be the set of all Ops c s.t. there is no implicit constraint
  // between a and c. It is easy to see that (a,b) is tight implies C(a) =
  // C(b), but C(a) = C(b) does not imply (a,b) is tight.
  std::vector<std::array<OpAddress, 2>> getTightPairs() const;

  // Starting from "a" and proceeding through Op outputs, find a chain of
  // tightly pairs Op. The returned vector may be the singleton, {a}, if it is
  // not tightly coupled to an output.
  std::vector<OpAddress> tightChainFrom(OpAddress a) const;

  // All constraints which are in this Graph, but not in "rhs"
  std::vector<std::vector<OpAddress>>
  constraintDiff(const std::vector<std::vector<OpAddress>> &rhs) const;

  std::vector<std::vector<OpAddress>> constraintDiff(const Graph &rhs) const {
    return constraintDiff(rhs.getForwardEdges());
  }

private:
  // Return true if there are no linked Ops which would be disconnected by a
  // shift of Ops
  bool isLinkPreserving(ScheduleIndex start0,
                        ScheduleIndex start1,
                        int nToShift) const;

  template <typename T>
  ScheduleIndex getExtremaIndexWithNonUniqueSolution() const;

  // kahn will merge any links then call linkless khan.
  void kahn(KahnTieBreaker, uint32_t kahnSeed);
  void linklessKahn(KahnTieBreaker, uint32_t kahnSeen);

  void setScheduleFromMergedChild(const OpMerged &merged);

  void removeConstraint(OpAddress before, OpAddress after);

  void confirmShiftAndCost(ScheduleIndex start0,
                           int nToShift,
                           const ShiftAndCost &shiftAndCost,
                           MinSumLivenessAlgo algo) const;

  // The first external consumer of an Op in the range [start, nToShift)
  ScheduleIndex getFirstConsumer(ScheduleIndex start, int nToShift) const;

  // The last external producer of an Op in the range [start, nToShift)
  ScheduleIndex getLastProducer(ScheduleIndex start, int nToShift) const;

  void applyChange(const ScheduleChange &);

  // Note, ALL vectors are maintained sorted in Graph and Op

  // unchanged after initialization: never updated
  std::vector<Op> allOps;
  std::vector<Alloc> allAllocs;

  // updated EVERY time the schedule changes
  std::vector<OpAddress> schToOp;
  std::vector<ScheduleIndex> opToSch;
  std::vector<std::vector<ScheduleIndex>> allocToSch;
  std::vector<std::vector<AllocAddress>> schToAllocs;
  std::vector<std::vector<ScheduleIndex>> opToInSch;
  std::vector<std::vector<ScheduleIndex>> opToOutSch;
  std::vector<int> nCanFwd;
  std::vector<int> nCanBwd;
  std::vector<bool> susceptible;

  // not updated every time the schedule changes
  std::vector<AllocWeight> schToLiveness;

  std::vector<AllocWeight> getRippleCosts(ScheduleIndex start0,
                                          int nToShift,
                                          int sign,
                                          int nCostsToCompute,
                                          int dirOffset) const;

  std::vector<AllocWeight>
  getFwdRippleCosts(ScheduleIndex start, int nToShift, int firstExtCon) const;

  std::vector<AllocWeight> getBwdRippleCosts(ScheduleIndex start0,
                                             int nToShift,
                                             int lastExtProd) const;

  ShiftAndCost getBestShiftRippleAlgo(const ScheduleIndex start,
                                      const int nToShift) const;

  ShiftAndCost getBestShiftSimpleAlgo(const ScheduleIndex start,
                                      const int nToShift) const;

  AllocWeight getShiftCost(ScheduleIndex start0,
                           ScheduleIndex start1,
                           int nToShift,
                           const Alloc &) const;

  std::vector<AllocAddress> getAllocAddresses(ScheduleIndex start,
                                              ScheduleIndex end) const;

  // collect, sort, and make unique, all outputs of Ops with ScheduleIndices
  // in [start, end).
  std::vector<OpAddress> getAllOutsInRange(ScheduleIndex start,
                                           ScheduleIndex end) const;

  // collect, sort, and make unique, all inputs of Ops with ScheduleIndices
  // in [start, end).
  std::vector<OpAddress> getAllInsInRange(ScheduleIndex start,
                                          ScheduleIndex end) const;

  std::vector<AllocWeight> getDeltaLiveness() const;

  void setSchToLiveness();

  void setOpToInSch(OpAddress);

  void setOpToOutSch(OpAddress);

  void setAllocToSch(AllocAddress);

  void setCanCan(int nToShift);

  void updateCanCan(int oldNToShift, int newNToShift);

  void updateNCanFwds(int nToShift,
                      int x0,
                      int o1,
                      const std::vector<OpAddress> &producersTouched);

  void updateNCanBwds(int nToShift,
                      int x0,
                      int o1,
                      const std::vector<OpAddress> &consumersTouched);

  // Susceptible Ops are Ops in (out) the range [rangeStart, rangeEnd] which
  // have a dependency outside the range
  void updateSusceptible(ScheduleIndex rangeStart, ScheduleIndex rangeEnd);

  // TODO(T14827) for multithreading, need one of these scratchpads per thread
  mutable std::vector<TrackEntry> rippleScratch;

  bool isFinalized{false};
  bool isInitialized{false};

  // The unique OpAddresses of Ops which have a forward link
  std::vector<OpAddress> opsWithFwdLinks;

public:
  // return all Ops which have the same ins as a. "a" is an element of the
  // returned vector
  std::vector<OpAddress> getIdenticalIns(OpAddress a) const;

  std::vector<std::vector<OpAddress>> getForwardEdges() const;

  // Combine all linked Ops form a sets of isolated chains
  std::vector<std::vector<OpAddress>> getLinkChains() const;

  // Combine all tight Op pairs to form sets of isolated chains
  std::vector<std::vector<OpAddress>> getTightChains() const;

  bool hasAtLeastOneLink() const { return !opsWithFwdLinks.empty(); }

  // insert a proxy Op, which is constrained to be scheduled very early, and
  // 1 Alloc, which must be live for "proxy" and Op "a". This attracts "a"
  // towards the beginning of the schedule. The Allocs' AllocWeights, which
  // determines the force of attraction of "a" to the beginning of the
  // schedule, determined by "relativeLexico" and "stepSize".
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

private:
  // Insert constraints and links which can be proven to satisfy at least one
  // globally minimizing schedule. These constraints accelerate the annealing
  // algorithm by reducing its search space.
  void
  applyTransitiveClosureOptimizations(const TransitiveClosureOptimizations &);

  transitiveclosure::TransitiveClosure transitiveClosure{{}};
  // The lowest change in liveness across all schedules, for each Op
  std::vector<AllocWeight> lowerBoundChange;
  // The highest change in liveness across all schedules, for each Op
  std::vector<AllocWeight> upperBoundChange;

  void initializeTransitiveClosure();

  // Incrementally update the TransitiveClosure of this Graph. Note that the
  // TransitiveClosure can be initialized with
  // updateTransitiveClosure(getForwardEdges()), but it is less efficient than
  // calling initializeTransitiveClosure().
  void
  updateTransitiveClosure(const std::vector<std::vector<OpAddress>> &nEdges);

  void finalizeTransitiveClosure();
  bool linkTightDrops();
  bool linkCloseTightPairs();
  bool constrainWeightSeparatedGroups();
  void processWeightSeparatedIdenticalIns(
      const std::vector<OpAddress> &opsWithIdenticalIns,
      std::vector<std::array<OpAddress, 2>> &cons) const;
  bool constrainParallelChains();
  bool slideLinks();
  void insertStartAttractorsAssert0(uint64_t, uint64_t) const;

  // Implements the isSchedulable algorithm assuming the graph has no links.
  bool linklessIsSchedulable() const;
};

std::ostream &operator<<(std::ostream &ost, const Graph &x);

std::ostream &operator<<(std::ostream &ost, const ShiftAndCost &x);

std::ostream &operator<<(std::ostream &ost, const ScheduleChange &x);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
