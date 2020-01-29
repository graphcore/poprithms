#ifndef POPRITHMS_SCHEDULE_ANNEAL_GRAPH
#define POPRITHMS_SCHEDULE_ANNEAL_GRAPH

#include <array>
#include <map>
#include <set>
#include <vector>
#include <poprithms/schedule/anneal/alloc.hpp>
#include <poprithms/schedule/anneal/allocweight.hpp>
#include <poprithms/schedule/anneal/annealusings.hpp>
#include <poprithms/schedule/anneal/op.hpp>
#include <poprithms/schedule/anneal/schedulechange.hpp>
#include <poprithms/schedule/anneal/shiftandcost.hpp>
#include <poprithms/schedule/anneal/trackentry.hpp>

// Design of the schedule annealing algorithm
// -------------------------------------------
// - store all schedule dependant information in the Graph class, not
//   the Op or the Alloc classes. With this decision, Ops and Allocs will
//   never be updated once the annealing begins
//
// - make the search algorithm for updates as fast as possible, at the expense
//   of the update algorithm. This because (1) finding swaps is easily
//   parallelisable and (2) updates are few and far between, especially at
//   later iterations of the algorithm, so most time is spent searching for
//   swaps

// TODO(T14827) Parallelize the search for energy reducing swaps. Suggestions:
// What the best approach is depends on whether we require the algorithm to be
// deterministic. Assuming that we do, this is what I propose (for the search)
// a vector of indices to process, toProcess, and a nextIndex, initialized to
// 0 Each thread, when ready, gets nextIndex and increments it by 1.
//
// It processes its index, and if there an improvement, requests all searching
// to halt. When all threads are finished their index, take the lowest index
// improves, call it updateIndex. Apply update, and reset nextIndex to
// updateIndex

namespace poprithms {
namespace schedule {
namespace anneal {

// Algorithms give exactly same results, RIPPLE is just much faster
enum class MinSumLivenessAlgo { SIMPLE, RIPPLE };

// The algorithm is initialized with a single run of Khan's algorithm, the
// tie-breaker does not make much difference to overall performance of the
// algorithm but GREEDY means slightly fewer shifts are required when
// annealing starts
enum class KhanTieBreaker { RANDOM, GREEDY };

class Graph {
public:
  // The Graph is grown incrementally with these 4 functions:

  // Create an Alloc
  AllocAddress insertAlloc(AllocWeight w);
  AllocAddress insertAlloc(double w) { return insertAlloc({w, 0}); }

  // Create an Op:
  OpAddress insertOp(const std::string &dbString);

  // Register that "aa" must be live when "oa" executes
  void insertOpAlloc(OpAddress oa, AllocAddress aa);

  // Register that "before" must execute before "after"
  void insertConstraint(OpAddress before, OpAddress after);

  // The above 4 methods are combined in a convenience method:
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
  OpAddress insertOp(std::initializer_list<OpAddress> befores,
                     std::initializer_list<AllocAddress> allocs,
                     const std::string &dbString) {
    return insertOp(std::vector<OpAddress>(befores),
                    std::vector<AllocAddress>(allocs),
                    dbString);
  }

  void append(std::ostream &ost) const;

  const std::vector<Op> &getOps() const { return allOps; }
  const Op &getOp(OpAddress address) const { return allOps[address]; }
  uint64_t nOps() const { return allOps.size(); }
  int nOps_i32() const { return static_cast<int>(nOps()); }

  const std::vector<Alloc> &getAllocs() const { return allAllocs; }
  const Alloc &getAlloc(AllocAddress address) const {
    return allAllocs[address];
  }
  uint64_t nAllocs() const { return getAllocs().size(); }
  std::string getLivenessString() const;

  // to be called once, when growing of Graph is complete
  // greedy 20 % faster than pure random in some experiments
  void initialize(KhanTieBreaker    = KhanTieBreaker::GREEDY,
                  uint32_t khanSeed = 1011);

  // Should be called once after the final call to a growing member. Sorts
  // certain Op member ids to accelerate the annealing algorithm
  void finalize();
  bool isSchedulable() const;

  // verify that all graph connections are sensible
  void assertCorrectness() const;

  static bool defaultDebug() { return false; }
  static uint32_t defaultSeed() { return 1011; }
  static double defaultPStayPut() { return 10.0; }
  static double defaultPHigherFallRate() { return 2.0; }
  static double defaultPClimb() { return 1.0; }
  static bool defaultLogging() { return true; }
  static double defaultTimeLimitSeconds() { return 1e9; }
  static int64_t defaultSwapLimitCount() { return static_cast<int64_t>(1e9); }

  // All Ops which thus far do not have any input dependencies
  std::vector<OpAddress> getInputOps() const;

  // definition of a "round":
  // one iteration through all Ops to search for,
  // and possibly apply, improvements
  //
  // After each round with at least 1 improvement, the
  // algorithm chooses between 3 options:
  //
  // a) stay with current nToShift.
  //
  // b) choose between nToShift = 1 and current nToShift.
  // The choice is made based on whether the current shift
  // has the best recorded "sum liveness decrease" per second.
  // Note that this dependant on time of execution makes this
  // potentially non-deterministic if (b) is possible.
  //
  // c) increase nToShift, with probability pClimb.
  //
  // probabilities for a, b, c are:
  // a) pStayPut
  // b) pHigherFallRate
  // c) pClimb
  //
  // Other arguments are
  // algo : RIPPLE (recommended) or SIMPLE (slow) : identical scheduling but
  // RIPPLE uses tricks to make it fast
  //
  // debug : compares "algo" above to SIMPLE to confirm agreement, and checks
  // state of graph edges at each iteration. This makes execution slow
  //
  // seed : the algorithm
  //   1) randomly shuffles op indices in each round
  //   2) randomly chooses between a, b, c above.
  //
  // logging : log the choice between a,b,c at each round

  void
  minSumLivenessAnneal(MinSumLivenessAlgo algo = MinSumLivenessAlgo::RIPPLE,
                       bool debug              = defaultDebug(),
                       uint32_t seed           = defaultSeed(),
                       double pStayPut         = defaultPStayPut(),
                       double pHigherFallRate  = defaultPHigherFallRate(),
                       double pClimb           = defaultPClimb(),
                       bool logging            = defaultLogging(),
                       double timeLimitSeconds = defaultTimeLimitSeconds(),
                       int64_t swapLimitCount  = defaultSwapLimitCount());

  void minSumLivenessAnneal(const std::map<std::string, std::string> &);

  AllocWeight getMaxLiveness() const;

  AllocWeight getSumLiveness() const;

  AllocWeight scheduleToLiveness(ScheduleIndex i) const {
    return schToLiveness[i];
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
    return schToAllocs[i];
  }

  // schedule indices of an ops inputs, sorted
  const std::vector<ScheduleIndex> &opToInSchedule(OpAddress a) const {
    return opToInSch[a];
  }

  // schedule indices of an ops output, sorted
  const std::vector<ScheduleIndex> &opToOutSchedule(OpAddress a) const {
    return opToOutSch[a];
  }

  // any Allocs which are first used at a schedule index
  const std::vector<AllocAddress> &
  scheduleToAllocFirsts(ScheduleIndex i) const {
    return schToAllocFirsts[i];
  }

  // any Allocs which are last used at a schedule index
  const std::vector<AllocAddress> &
  scheduleToAllocFinals(ScheduleIndex i) const {
    return schToAllocFinals[i];
  }
  int getNCanFwd(ScheduleIndex i) const {
    return nCanFwd[static_cast<uint64_t>(i)];
  }
  int getNCanBwd(ScheduleIndex i) const {
    return nCanBwd[static_cast<uint64_t>(i)];
  }

  const std::vector<OpAddress> &getScheduleToOp() const { return schToOp; }

  // The following are convenience functions:

  // Ops in "bins" must execute in increasing bin index. For example, if a \in
  // bins[0] and b \in bins[1], then a must execute before b
  void insertBinConstraints(const std::vector<std::vector<OpAddress>> &bins,
                            const std::string &opPrefix);

  // pairs a,b \in "pairs" should be executed as close to each other as
  // possible, with "gravitational force" w
  void insertAttractions(const std::vector<std::array<OpAddress, 2>> &pairs,
                         AllocWeight w);

  bool operator==(const Graph &rhs) const {
    return allOps == rhs.allOps && allAllocs == rhs.allAllocs;
  }

  // A pair of Ops (a,b) is defined to be a "tight pair" if
  // 1) b is the only output of a,
  // 2) a is the only input of b.
  // Let C(a) be the set of all Ops c s.t. there is no implicit constraint
  // between a and c. It is easy to see that (a,b) is tight implies C(a) =
  // C(b), but C(a) = C(b) does not imply (a,b) is tight.
  std::vector<std::array<OpAddress, 2>> getTightPairs() const;

private:
  template <typename T>
  ScheduleIndex getExtremaIndexWithNonUniqueSolution() const;

  void khan(KhanTieBreaker, uint32_t khanSeed);

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
  std::vector<std::vector<AllocAddress>> schToAllocFirsts;
  std::vector<std::vector<AllocAddress>> schToAllocFinals;
  std::vector<int> nCanFwd;
  std::vector<int> nCanBwd;

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

  // TODO(T14827) for multithreading, need one of these scratchpads per thread
  mutable std::vector<TrackEntry> rippleScratch;

  bool isFinalized{false};

public:
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

    // For each Op "a" \in opAddresses, the size of the attracting Alloc is
    // determined by the corresponding priority in "priorities"
    assert(opAddresses.size() == priorities.size());

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
};

std::ostream &operator<<(std::ostream &ost, const Graph &x) {
  x.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const ShiftAndCost &x) {
  x.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const ScheduleChange &x) {
  x.append(ost);
  return ost;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
