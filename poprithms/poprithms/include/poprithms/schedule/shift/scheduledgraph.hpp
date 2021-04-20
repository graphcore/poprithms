// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SCHEDULEDGRAPH
#define POPRITHMS_SCHEDULE_SHIFT_SCHEDULEDGRAPH

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/kahntiebreaker.hpp>
#include <poprithms/schedule/shift/rotationalgo.hpp>
#include <poprithms/schedule/shift/rotationtermination.hpp>
#include <poprithms/schedule/shift/settings.hpp>
#include <poprithms/schedule/shift/shiftandcost.hpp>
#include <poprithms/schedule/shift/solutioncache.hpp>
#include <poprithms/schedule/shift/trackentry.hpp>
#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class ScheduleChange;

/**
 * A Graph with a fixed schedule. It is constructed from a (unscheduled) Graph
 * and several settings which control the scheduling algorithm.
 *
 * The core optimization algorithm implemented for this class attempts to
 * minimize the sum of the livenesses of the Allocs, where an Alloc is live
 * from the first to last of its Ops' schedule indices.
 *
 * For example, if an Alloc 'a' has Ops {'b','c','d'} which require it to be
 * live, and the schedule indices of 'b','c', and 'd' are 5,8 and 11
 * respectively, then 'a' is live for a duration of 11 - 5 + 1 = 7. Further
 * information is available in the notes directory of poprithms.
 *
 * This optimization algorithms is referred to as "rotation".
 * */

class ScheduledGraph {
public:
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
   *
   * \param ktb The method used to choose an Op from a set which are ready to
   *            be scheduled
   *
   * \param tco The set of optimizations to apply to the Graph, to accelerate
   *            the min-sum-liveness algorithm. These optimizations insert
   *            constraints and links between Ops which all sum-liveness
   *            minimizing schedules satisfy.
   *
   * \param algo Implementation of the rotation algorithm to use.
   *
   * \param debug Compares algo (above) to SIMPLE to confirm agreement, and
   *              checks state of graph edges at each iteration. debug=true
   *              makes execution slow.
   *
   * \param seed The random seed is used when the KahnTieBreaker is Random, as
   *             well as in the rotation optimization algorithm.
   *   */

  ScheduledGraph(
      Graph &&,
      KahnTieBreaker ktb                 = Settings::defaultKahnTieBreaker(),
      TransitiveClosureOptimizations tco = Settings::defaultTCOs(),
      RotationTermination rt         = Settings::defaultRotationTermination(),
      RotationAlgo algo              = Settings::defaultRotationAlgo(),
      uint32_t seed                  = Settings::defaultSeed(),
      DebugMode dm                   = Settings::defaultDebugMode(),
      const SolutionCache *readCache = nullptr,
      SolutionCache *writeCache      = nullptr);

  ScheduledGraph(Graph &&,
                 const Settings &,
                 const SolutionCache * = nullptr,
                 SolutionCache *       = nullptr);

  ScheduledGraph(Graph &&,
                 const std::map<std::string, std::string> &,
                 const SolutionCache * = nullptr,
                 SolutionCache *       = nullptr);

  static bool isSchedulable(const Graph &);

  /** verify that all graph connections are valid, if not throw error */
  void assertCorrectness() const;

  std::string getLivenessString() const;

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
   * \return Vector such that position i is the OpAddress of the i^th op in
   *         the internal schedule.
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

  const Graph &getGraph() const { return graph; }

  int32_t nOps_i32() const { return graph.nOps_i32(); }
  uint64_t nOps() const { return graph.nOps(); }
  uint64_t nAllocs() const { return graph.nAllocs(); }
  const Op &getOp(OpAddress a) const { return graph.getOp(a); }
  const Alloc &getAlloc(AllocAddress a) const { return graph.getAlloc(a); }
  std::vector<std::vector<OpAddress>> getForwardEdges() const {
    return graph.getForwardEdges();
  }

  bool isFromCache() const { return fromCache; }

private:
  void
  initialize(KahnTieBreaker, uint32_t seed, TransitiveClosureOptimizations);

  void
  greedyRotate(RotationAlgo, DebugMode, uint32_t seed, RotationTermination);

  // Return true if there are no linked Ops which would be disconnected by a
  // shift of Ops
  bool isLinkPreserving(ScheduleIndex start0,
                        ScheduleIndex start1,
                        int nToShift) const;

  template <typename T>
  ScheduleIndex getExtremaIndexWithNonUniqueSolution() const;

  // kahn will merge any links then call linkless khan.
  static std::vector<OpAddress>
  kahn(const Graph &, KahnTieBreaker, uint32_t seed);

  static std::vector<OpAddress>
  linklessKahn(const Graph &, KahnTieBreaker, uint32_t kahnSeen);

  static std::vector<OpAddress>
  getScheduleFromMergedChild(const Graph::OpMerged &merged,
                             const std::vector<OpAddress> &childSchedule);

  void confirmShiftAndCost(ScheduleIndex start0,
                           int nToShift,
                           const ShiftAndCost &shiftAndCost,
                           RotationAlgo algo) const;

  // The first external consumer of an Op in the range [start, nToShift)
  ScheduleIndex getFirstConsumer(ScheduleIndex start, int nToShift) const;

  // The last external producer of an Op in the range [start, nToShift)
  ScheduleIndex getLastProducer(ScheduleIndex start, int nToShift) const;

  void applyChange(const ScheduleChange &);

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

  // bool isInitialized{false};

  // Insert constraints and links which can be proven to satisfy at least one
  // globally minimizing schedule. These constraints accelerate the shifting
  // algorithm by reducing its search space.
  void
  applyTransitiveClosureOptimizations(const TransitiveClosureOptimizations &);

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

  // Implements the isSchedulable algorithm assuming the graph has no links.
  static bool linklessIsSchedulable(const Graph &);

  // not const: might change!
  Graph graph;

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

  transitiveclosure::TransitiveClosure transitiveClosure{{}};
  // The lowest change in liveness across all schedules, for each Op
  std::vector<AllocWeight> lowerBoundChange;
  // The highest change in liveness across all schedules, for each Op
  std::vector<AllocWeight> upperBoundChange;

  bool fromCache{false};
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
