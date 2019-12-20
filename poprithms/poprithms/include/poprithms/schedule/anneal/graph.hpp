#include <map>
#include <vector>
#include <poprithms/schedule/anneal/alloc.hpp>
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

// How to parallelize. TODO(jn)
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
  // The Graph is grown incrementally with these 2 functions:
  // 1)
  OpAddress insertOp(const std::vector<OpAddress> &ins,
                     const std::vector<AllocAddress> &,
                     const std::string &_debugString_);
  // 2)
  AllocAddress insertAlloc(AllocWeight);

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

  // verify that all graph connections are sensible
  void assertCorrectness() const;

  static bool defaultDebug() { return false; }
  static uint32_t defaultSeed() { return 1011; }
  static double defaultPStayPut() { return 10.0; }
  static double defaultPHigherFallRate() { return 2.0; }
  static double defaultPClimb() { return 1.0; }
  static bool defaultLogging() { return true; }

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
                       bool logging            = defaultLogging());

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

  // any allocations which are first used at a schedule index
  const std::vector<AllocAddress> &
  scheduleToAllocFirsts(ScheduleIndex i) const {
    return schToAllocFirsts[i];
  }

  // any allocations which are last used at a schedule index
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

private:
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

  // TODO(jn) for multithreading, need one of these scratchpads per thread
  mutable std::vector<TrackEntry> rippleScratch;
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
