// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

#include <schedule/shift/allocsimplifier.hpp>
#include <schedule/shift/error.hpp>
#include <schedule/shift/greedykahn.hpp>
#include <schedule/shift/transitiveclosureoptimizer.hpp>
#include <schedule/shift/updatefromfirstfinal.hpp>

#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/shift/filteredschedule.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/shift/schedulecache.hpp>
#include <poprithms/schedule/shift/schedulechange.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

// Design of the schedule shifting algorithm
// -------------------------------------------
//
// - make the search algorithm for updates as fast as possible, at the expense
//   of the update algorithm. This because (1) finding swaps is easily
//   parallelizable and (2) updates are few and far between, especially at
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

class Settings;

namespace {

std::vector<OpAddress>
kahn(const Graph &graph, const KahnDecider &kd, const uint32_t seed) {

  auto getSchedule = [&graph, &kd, &seed](vanilla::ErrorIfCycle eic) {
    const auto links = graph.getFwdLinks();

    switch (kd.kahnTieBreaker()) {
    case KahnTieBreaker::FIFO: {
      return vanilla::Scheduler<uint64_t, double>::filo(
          graph.getForwardEdges(),
          kd.priorities(),
          links,
          eic,
          vanilla::VerifyEdges::No);
    }
    case KahnTieBreaker::GREEDY: {

      std::vector<AllocWeight> allocSizes;
      std::vector<std::vector<uint64_t>> allocsToOps;
      for (auto x : graph.getAllocs()) {
        allocSizes.push_back(x.getWeight());
        allocsToOps.push_back(x.getOps());
      }

      return greedyKahn(graph.getFwdEdges_u64(),
                        kd.priorities(),
                        links,
                        allocSizes,
                        allocsToOps,
                        eic,
                        vanilla::VerifyEdges::No);
    }
    case KahnTieBreaker::RANDOM: {
      return vanilla::Scheduler<uint64_t, double>::random(
          graph.getForwardEdges(),
          kd.priorities(),
          links,
          seed,
          eic,
          vanilla::VerifyEdges::No);
    }
    default: {
      throw error("Unrecognised KahnTieBreaker.");
    }
    }
  };

  auto sch = getSchedule(vanilla::ErrorIfCycle::No);

  if (sch.size() != graph.nOps()) {

    log().info("Failed to schedule all Ops, obtaining summary.");

    // if the graph has links in it, we throw an error directly from the
    // vanilla scheduler (which provides better diagnostics in this case).
    if (graph.getOpsWithFwdLinks().size() != 0) {
      auto sch2 = getSchedule(vanilla::ErrorIfCycle::Yes);
      // this point should not be reached.
      throw error(
          "Failed to schedule graph with links in it, cycle detected.");
    }

    std::vector<std::string> dbs;
    dbs.reserve(graph.nOps());
    for (uint64_t i = 0; i < graph.nOps(); ++i) {
      dbs.push_back(graph.getOp(i).getDebugString());
    }

    std::ostringstream oss;
    oss << "Only " << sch.size() << " of " << graph.nOps()
        << " were scheduled, there is a cycle in the Graph."
        << " The strongly connected components with cycles, "
        << "in topological order, are:"
        << scc::getSummary(graph.getFwdEdges_u64(),
                           dbs,
                           scc::IncludeCyclelessComponents::No);

    throw error(oss.str());
  }

  return sch;
}

template <typename T>
void rotate(T &t, ScheduleIndex x0, ScheduleIndex o0, ScheduleIndex o1) {
  auto t0 = std::next(t.begin(), x0);
  auto t1 = std::next(t.begin(), o0);
  auto t2 = std::next(t.begin(), o1);
  std::rotate(t0, t1, t2);
}
// 1) For all indices i in [start, end),
//    append f(i) to an output vector. Then,
// 2) Sort the output element vector, using std::sort. Then,
// 3) Remove duplicates.
//
//   F f maps a ScheduleIndex to a container of A's
//   A is some address type (AllocAddress or OpAddress)
template <typename A, class F>
std::vector<A>
getInRangeStdSort(const ScheduleIndex start, const ScheduleIndex end, F &&f) {

  std::vector<A> addresses;
  // No obvious size to reserve for addresses, so not performing a reserve.

  for (ScheduleIndex i = start; i < end; ++i) {
    for (A a : f(i)) {
      addresses.push_back(a);
    }
  }

  // At this point, addresses is not sorted and may contain duplicated.
  // Options are to 1) insert into std::set and return set's range, or 2)
  // std::sort and the use std::unique. Some experiments showed that 2 is
  // faster.

  std::sort(addresses.begin(), addresses.end());
  auto last = std::unique(addresses.begin(), addresses.end());
  addresses.erase(last, addresses.cend());
  return addresses;
}

// Equivalent to getInRangeStdSort, but faster in certain cases.
// 1) Create a bool vector of size nBuckets, all false.
// 2) For all indices i in [start, end), set all bools at indices in
//    f(i) to true.
// 3) return all indices which are true.
template <typename A, class F>
std::vector<A> getInRangeBucketSort(const ScheduleIndex start,
                                    const ScheduleIndex end,
                                    const uint64_t nBuckets,
                                    F &&f) {
  // Example:
  //
  // start    = 2
  // end      = 6
  // nBuckets = 5
  //
  //
  // 0 1 2 3 4 5 6 7     (schedule indices)
  //     [       )       [start, end) range to iterate over
  //     | | | |
  //     v v v v
  //     0 3 0 0         (values from call to f)
  //     1   3
  //
  //                           0     1      2     3      4
  //                           x
  //                           x                  x
  //                           x     x            x
  // buckets after filling : [true, true, false, true, false]
  //

  std::vector<bool> bins(nBuckets, false);
  uint64_t nFullBins{0};
  for (ScheduleIndex i = start; i < end; ++i) {
    for (auto a : f(i)) {
      if (!bins[a]) {
        ++nFullBins;
      }
      bins[a] = true;
    }
  }
  std::vector<A> addresses;
  addresses.reserve(nFullBins);
  for (uint64_t i = 0; i < bins.size(); ++i) {
    if (bins[i]) {
      addresses.push_back(i);
    }
  }
  return addresses;
}

std::string getRotationAlgoString(const RotationAlgo algo) {
  switch (algo) {
  case (RotationAlgo::SIMPLE): {
    return "Simple";
  }
  case (RotationAlgo::RIPPLE): {
    return "Ripple";
  }
  }
  throw error("Unrecognised RotationAlgo");
}

std::ostream &operator<<(std::ostream &os, const std::vector<uint64_t> &v) {
  poprithms::util::append(os, v);
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<int> &v) {
  poprithms::util::append(os, v);
  return os;
}

constexpr const char *const spaces = "         ";

// comment-I (cited in code)
// ------------
// if it's not possible to move more than nToShift - 1, then all
// shifts which can be considered here would have been considered at a
// lower level, or will be considered at a lower level at a later point

template <class ForwardIt, class T>
ForwardIt
custom_lower_bound(ForwardIt first, ForwardIt last, const T &value) {
  return std::lower_bound(first, last, value);

  // while below is observed to be marginally faster in experiments, I'm
  // deciding to stick with std::lower_bound (which has better complexity)
  /*
  ForwardIt it = first;
  while(it != last && *it < value){
    std::advance(it, 1);
  }
  return it;
  */
}

// where:
//   F f maps a ScheduleIndex to a container of A's
//   A is some address type (AllocAddress or OpAddress)
template <typename A, class F>
std::vector<A> getInRange(const ScheduleIndex start,
                          const ScheduleIndex end,
                          const uint64_t nBuckets,
                          F &&f) {

  // Choose between std::sort and bucket sort:
  //
  // Let K be the expected number of elements returned from calling f.
  // Let S = K*(end - start).
  // Complexity of bucket     = O(nBuckets + S)
  // Complexity of std::sort  = O(S*log(S)).
  //
  // So we use bucket if,
  //   O(nBuckets + S) < O(S*log(S))
  //
  // Ignoring complexity constants,
  //   nBuckets < S*log(S)

  const auto S          = 2 * (end - start + 1);
  const auto SlogS      = S * log2(S);
  const auto useBuckets = nBuckets < SlogS;
  if (useBuckets) {
    return getInRangeBucketSort<A>(start, end, nBuckets, f);
  } else {
    return getInRangeStdSort<A>(start, end, f);
  }
}

} // namespace

std::ostream &operator<<(std::ostream &ost, const ScheduleChange &x) {
  x.append(ost);
  return ost;
}

void ScheduledGraph::initialize(const KahnDecider &kd,
                                const uint32_t seed,
                                const TransitiveClosureOptimizations tco,
                                const ISummaryWriter &summaryWriter) {

  const auto stopwatch = timeLogger().scopedStopwatch("initialize");

  std::ostringstream oss;
  oss << "Graph::initialize() entered for Graph with " << nOps() << " Ops, "
      << graph.nAllocs() << " Allocs, " << graph.nConstraints()
      << " constraints. ";
  log().info(oss.str());

  TransitiveClosureOptimizer::apply(tco, graph, timeLogger());

  //
  // schToOp. Vanilla run of Kahn's O(E) algorithm, random tie-breaks
  schToOp = kahn(graph, kd, seed);
  summaryWriter.writeInitialSchedule(schToOp);

  //
  // opToSch
  opToSch.clear();
  opToSch.reserve(nOps());
  opToSch.resize(nOps_i32());

  for (ScheduleIndex i = 0; i < nOps_i32(); ++i) {
    opToSch[scheduleToOp(i)] = i;
  }

  //
  // allocToSch
  allocToSch.clear();
  allocToSch.reserve(graph.nAllocs());
  allocToSch.resize(graph.nAllocs());

  for (AllocAddress allocAddress = 0; allocAddress < graph.nAllocs();
       ++allocAddress) {
    setAllocToSch(allocAddress);
  }

  //
  // schToAllocs
  schToAllocs.reserve(nOps());
  schToAllocs.clear();

  for (ScheduleIndex schedIndex = 0; schedIndex < nOps_i32(); ++schedIndex) {
    auto schedIndex_u64 = static_cast<uint64_t>(schedIndex);
    auto opAddress      = scheduleToOp(schedIndex);
    const auto &op      = getOp(opAddress);
    schToAllocs.push_back({});
    schToAllocs[schedIndex_u64].reserve(op.nAllocs());
    for (AllocAddress allocAddress : op.getAllocs()) {
      schToAllocs[schedIndex_u64].push_back(allocAddress);
    }
    std::sort(schToAllocs[schedIndex_u64].begin(),
              schToAllocs[schedIndex_u64].end());
  }

  //
  // opToInSch
  opToInSch.resize(nOps());
  for (OpAddress opAddress = 0; opAddress < nOps(); ++opAddress) {
    setOpToInSch(opAddress);
  }

  //
  // opToOutSch
  opToOutSch.resize(nOps());
  for (OpAddress opAddress = 0; opAddress < nOps(); ++opAddress) {
    setOpToOutSch(opAddress);
  }

  //
  // schToLiveness
  setSchToLiveness();

  //
  // nFwd, nBwd
  setCanCan(1);

  rippleScratch.resize(
      graph.nAllocs(),
      {-1, AllocWeight::negativeOne(), AllocWeight::negativeOne(), false});

  assertCorrectness();
}

void ScheduledGraph::setCanCan(int nToShift) {

  nCanFwd.clear();
  auto numNCan =
      static_cast<uint64_t>(std::max(0, nOps_i32() - nToShift + 1));
  nCanFwd.reserve(numNCan);
  nCanBwd.clear();
  nCanBwd.reserve(numNCan);

  for (ScheduleIndex i = 0; i < nOps_i32() - nToShift + 1; ++i) {
    // how far can the sub-schedule [i + 1, i + 1 + nToShift) move
    // backwards?
    nCanBwd.push_back(i - getLastProducer(i, nToShift) - 1);

    // can the sub-schedule [i, i + nToShift) move forwards?
    nCanFwd.push_back(getFirstConsumer(i, nToShift) - i - nToShift);
  }
}

void ScheduledGraph::updateCanCan(const int oldNToShift, const int n2s) {

  // bootstrapping off oldNToShift is significantly faster
  if (n2s - oldNToShift == 1) {

    // with an increase of 1 of nToShift, the number of possible starts
    // decreases by 1
    if (!nCanFwd.empty()) {
      nCanFwd.pop_back();
      nCanBwd.pop_back();
    }

    for (ScheduleIndex i = 0; i < nOps_i32() - n2s + 1; ++i) {
      auto i_u64          = static_cast<uint64_t>(i);
      auto firstOpAddress = scheduleToOp(i);
      auto finalOpAddress = scheduleToOp(i + n2s - 1);

      // update nCanBwd
      const auto &inSched = opToInSchedule(finalOpAddress);
      auto x = custom_lower_bound(inSched.cbegin(), inSched.cend(), i);
      if (x != inSched.cbegin()) {
        nCanBwd[i_u64] = std::min(nCanBwd[i_u64], i - 1 - *std::prev(x));
      }

      // update nCanFwd
      const auto &outSched = opToOutSchedule(firstOpAddress);
      if (i == nOps_i32() - n2s) {
        nCanFwd[i_u64] = 0;
      } else {
        nCanFwd[i_u64] = nCanFwd[i_u64 + 1];
        x = custom_lower_bound(outSched.cbegin(), outSched.cend(), i + n2s);
        if (x != outSched.cend()) {
          nCanFwd[i_u64] = std::min(nCanFwd[i_u64], *x - (i + n2s));
        }
      }
    }
  }

  // currently we don't bootstrap for other size changes, as increasing by 1
  // is the only change in the control algorithm (other than dropping to 1)
  else {
    setCanCan(n2s);
  }
}

std::vector<AllocWeight> ScheduledGraph::getDeltaLiveness() const {
  std::vector<AllocWeight> deltaLiveness(nOps() + 1, AllocWeight::zero());
  for (AllocAddress allocAddress = 0; allocAddress < nAllocs();
       ++allocAddress) {
    if (getAlloc(allocAddress).nOps() > 0) {
      auto w = getAlloc(allocAddress).getWeight();
      auto firstSched =
          static_cast<uint64_t>(allocToFirstSchedule(allocAddress));
      auto finalSched =
          static_cast<uint64_t>(allocToFinalSchedule(allocAddress));
      deltaLiveness[firstSched] += w;
      deltaLiveness[finalSched + 1] -= w;
    }
  }
  return deltaLiveness;
}

void ScheduledGraph::assertCorrectness() const {

  // 1
  // opToSch vs schToOp
  for (uint64_t i = 0; i < nOps(); ++i) {
    if (opToSch[schToOp[i]] != static_cast<ScheduleIndex>(i)) {
      throw error("OpToSch and schToOp do not agree at ScheduleIndex " +
                  std::to_string(i));
    }
    if (schToOp[static_cast<uint64_t>(opToSch[i])] != i) {
      throw error("OpToSch and schToOp do not agree for OpAddress " +
                  std::to_string(i));
    }
  }

  // opToInSch, opToOutSch
  for (auto opAddress = 0UL; opAddress < nOps(); ++opAddress) {
    for (auto inAddress : getOp(opAddress).getIns()) {
      if (std::find(opToInSch[opAddress].cbegin(),
                    opToInSch[opAddress].cend(),
                    opToSch[inAddress]) == opToInSch[opAddress].cend()) {
        throw error("opToInSch is incorrect");
      }
    }

    for (auto outAddress : getOp(opAddress).getOuts()) {
      if (std::find(opToOutSch[opAddress].cbegin(),
                    opToOutSch[opAddress].cend(),
                    opToSch[outAddress]) == opToOutSch[opAddress].cend()) {
        throw error("opToOutSch is incorrect");
      }
    }
  }

  // topological constraints
  for (uint64_t i = 0; i < nOps(); ++i) {
    for (auto j : getOp(schToOp[i]).getIns()) {
      if (opToSch[j] >= ScheduleIndex(i)) {
        std::ostringstream oss;
        oss << "\n\nTopological constrains not satisfied. ";
        oss << "Op " << schToOp[i] << " is scheduled at " << i
            << " and its input:schedule(s) are: \n";
        for (auto k : getOp(schToOp[i]).getIns()) {
          oss << " " << k << ":" << opToSch[k];
        }
        throw error(oss.str());
      }
    }
  }

  // links:
  for (const auto &op0 : graph.getOps()) {
    if (op0.hasForwardLink()) {
      auto op1Address = op0.getForwardLink();
      if (opToSchedule(op1Address) != opToSchedule(op0.getAddress()) + 1) {
        throw error("Link is not satisfied, failure in assertCorrectness");
      }
    }
  }
}

ShiftAndCost
ScheduledGraph::getBestShiftSimpleAlgo(const ScheduleIndex start0,
                                       const int nToShift) const {

  // sum over Allocs of allocWeight * ( liveness duration )
  auto getTotal = [this](const std::vector<ScheduleIndex> &o2s) {
    AllocWeight tot{0};
    for (const auto &alloc : graph.getAllocs()) {
      if (alloc.nOps() > 0) {
        ScheduleIndex i0 = nOps_i32();
        ScheduleIndex i1 = -1;
        for (auto opAddress : alloc.getOps()) {
          auto opIndex = o2s[opAddress];
          i0           = std::min(i0, opIndex);
          i1           = std::max(i1, opIndex);
        }
        auto totAlloc = (i1 - i0 + 1) * alloc.getWeight();
        tot += totAlloc;
      }
    }
    return tot;
  };

  auto currentTotal = getTotal(opToSch);

  auto bestTotal = currentTotal;
  int bestShift  = 0;

  // consider all changes start -> x where x in [s0, s1)
  auto s0 = getLastProducer(start0, nToShift) + 1;

  if (s0 != start0 - getNCanBwd(start0)) {
    throw error("Error detected in getBestShiftSimpleAlgo, unexpected "
                "getNCanBwd value");
  }

  // see comment-I for what this comment does
  if (getNCanBwd(start0) < nToShift) {
    s0 = start0;
  }

  auto s1 = getFirstConsumer(start0, nToShift) - nToShift + 1;
  if (s1 != start0 + getNCanFwd(start0) + 1) {
    throw error("Error detected in getBestShiftSimpleAlgo, unexpected "
                "getNCanFwd value");
  }

  // see comment-I for what this comment does
  if (getNCanFwd(start0) < nToShift) {
    s1 = start0;
  }

  for (ScheduleIndex start1 = s0; start1 < s1; ++start1) {
    auto newSchedToOp = schToOp;
    int a, b, c;
    if (start0 < start1) {
      a = start0;
      b = start0 + nToShift;
      c = start1 + nToShift;
    }

    else {
      a = start1;
      b = start0;
      c = start0 + nToShift;
    }

    std::rotate(std::next(newSchedToOp.begin(), a),
                std::next(newSchedToOp.begin(), b),
                std::next(newSchedToOp.begin(), c));

    std::vector<ScheduleIndex> newOpToSched(nOps());
    for (uint64_t i = 0; i < nOps(); ++i) {
      newOpToSched[newSchedToOp[i]] = static_cast<ScheduleIndex>(i);
    }

    auto newTotal = getTotal(newOpToSched);

    auto isLinkPreservingSwitcher = [this, &start0, &start1, nToShift]() {
      if (start0 < start1) {
        return isLinkPreserving(start0, start1, nToShift);
      } else {
        return isLinkPreserving(start1, start1 + nToShift, start0 - start1);
      }
    };

    if (newTotal < bestTotal) {
      if (isLinkPreservingSwitcher()) {
        bestTotal = newTotal;
        bestShift = start1 - start0;
      }
    }
  }

  return {bestShift, bestTotal - currentTotal};
}

AllocWeight ScheduledGraph::getMaxLiveness() const {
  if (schToLiveness.empty()) {
    throw error(
        "Call go getMaxLiveness, but schToLiveness has not yet been set");
  }
  return std::accumulate(
      schToLiveness.cbegin(),
      schToLiveness.cend(),
      static_cast<AllocWeight>(0),
      [](AllocWeight a, AllocWeight b) { return std::max(a, b); });
}

AllocWeight ScheduledGraph::getSumLiveness() const {
  if (schToLiveness.empty()) {
    throw error(
        "Call go getSumLiveness, but schToLiveness has not yet been set");
  }
  return std::accumulate(schToLiveness.cbegin(),
                         schToLiveness.cend(),
                         static_cast<AllocWeight>(0),
                         [](AllocWeight a, AllocWeight b) { return a + b; });
}

std::vector<AllocWeight>
ScheduledGraph::getBwdRippleCosts(ScheduleIndex start0,
                                  int nToShift,
                                  int lastExtProd) const {

  const int sign             = -1;
  const auto nCostsToCompute = start0 - lastExtProd - 1;
  const auto dirOffset       = 0;
  return getRippleCosts(start0, nToShift, sign, nCostsToCompute, dirOffset);
}

// this was the trickiest function to get right
std::vector<AllocWeight>
ScheduledGraph::getRippleCosts(const ScheduleIndex start0,
                               const int nToShift,
                               const int sign,
                               const int nCostsToCompute,
                               const int dirOffset) const {

  const auto boundEnd = nCostsToCompute + sign * start0 + 1;

  std::vector<AllocWeight> costs;
  costs.reserve(static_cast<uint64_t>(nCostsToCompute));

  // Cumulative cost for starts, increasing away from start0
  AllocWeight w{0};

  AllocWeight toIncrement{0};

  std::vector<AllocAddress> liveAllocAddresses;

  // the usual suspects
  auto x0 = start0;
  auto o0 = start0 + nToShift;

  // initialize registry and toIncrement
  auto initialAllocAddresses = getAllocAddresses(x0, o0);
  liveAllocAddresses.reserve(
      initialAllocAddresses.size() +
      static_cast<uint64_t>(std::abs(boundEnd - start0)));
  for (auto allocAddress : initialAllocAddresses) {
    const auto &schedInds = allocToSchedule(allocAddress);
    auto firstX =
        custom_lower_bound(schedInds.cbegin(), schedInds.cend(), x0);
    auto firstO       = custom_lower_bound(firstX, schedInds.cend(), o0);
    int isPre         = firstX != schedInds.cbegin();
    int isPost        = firstO != schedInds.cend();
    const auto wAlloc = getAlloc(allocAddress).getWeight();
    AllocWeight wIncr = sign * (isPre - isPost) * wAlloc;
    liveAllocAddresses.push_back(allocAddress);
    rippleScratch[allocAddress] = {start0, AllocWeight::zero(), wIncr, true};
    toIncrement += wIncr;
  }

  for (ScheduleIndex start1 = start0 + sign; sign * start1 < boundEnd;
       start1 += sign) {

    // for all allocations at the new final position of "o", check if seen
    // in "x" or existing "o" and remove all record on w and toIncrement
    const auto &start1Allocs = scheduleToAllocs(start1 + dirOffset);
    for (auto a : start1Allocs) {
      if (rippleScratch[a].live) {
        auto record = rippleScratch[a];
        w -= record.entryWeight;
        auto incrTime = sign * (start1 - record.entryTime) - 1;
        w -= incrTime * record.incrWeight;
        toIncrement -= record.incrWeight;
      }
    }

    w += toIncrement;

    // having removed  previous effect of allocs, insert up-to-date entries
    for (auto allocAddress : start1Allocs) {
      auto partCost =
          getShiftCost(start0, start1, nToShift, getAlloc(allocAddress));
      const auto &schedInds = allocToSchedule(allocAddress);
      const auto extremum   = (sign == -1 ? schedInds[0] : schedInds.back());

      // only in a special case will incrWeight be non-zero:
      // TODO(T14829) diagram explaining this special case.
      auto newIncr = AllocWeight::zero();
      auto post0 =
          custom_lower_bound(schedInds.cbegin(), schedInds.cend(), start0);
      if (post0 != schedInds.cend() && *post0 - start0 < nToShift &&
          extremum == start1 + dirOffset) {
        newIncr = getAlloc(allocAddress).getWeight();
      }

      if (!rippleScratch[allocAddress].live) {
        liveAllocAddresses.push_back(allocAddress);
      }
      rippleScratch[allocAddress] = {start1, partCost, newIncr, true};
      w += partCost;
      toIncrement += newIncr;
    }

    costs.push_back(w);
  }

  // clear the scratchpad for future runs
  for (auto x : liveAllocAddresses) {
    rippleScratch[x].live = false;
  }
  return costs;
}

void ScheduledGraph::setOpToInSch(const OpAddress opAddress) {
  opToInSch[opAddress].reserve(getOp(opAddress).nIns());
  opToInSch[opAddress].clear();
  for (OpAddress inAddress : getOp(opAddress).getIns()) {
    opToInSch[opAddress].push_back(opToSch[inAddress]);
  }
  std::sort(opToInSch[opAddress].begin(), opToInSch[opAddress].end());
}

void ScheduledGraph::setOpToOutSch(const OpAddress opAddress) {
  opToOutSch[opAddress].reserve(getOp(opAddress).nOuts());
  opToOutSch[opAddress].clear();
  for (OpAddress outAddress : getOp(opAddress).getOuts()) {
    opToOutSch[opAddress].push_back(opToSch[outAddress]);
  }
  std::sort(opToOutSch[opAddress].begin(), opToOutSch[opAddress].end());
}

void ScheduledGraph::setAllocToSch(const AllocAddress allocAddress) {
  allocToSch[allocAddress].reserve(getAlloc(allocAddress).nOps());
  allocToSch[allocAddress].clear();
  for (OpAddress opAddress : getAlloc(allocAddress).getOps()) {
    allocToSch[allocAddress].push_back(opToSch[opAddress]);
  }
  std::sort(allocToSch[allocAddress].begin(), allocToSch[allocAddress].end());
}

std::vector<AllocAddress>
ScheduledGraph::getAllocAddresses(const ScheduleIndex start,
                                  const ScheduleIndex end) const {
  const auto f = [this](ScheduleIndex i) -> auto & {
    return scheduleToAllocs(i);
  };
  return getInRange<AllocAddress>(start, end, nAllocs(), f);
}

std::vector<OpAddress>
ScheduledGraph::getAllInsInRange(const ScheduleIndex start,
                                 const ScheduleIndex end) const {
  const auto f = [this](ScheduleIndex i) -> auto & {
    return getOp(scheduleToOp(i)).getIns();
  };
  return getInRange<OpAddress>(start, end, nOps(), f);
}

std::vector<OpAddress>
ScheduledGraph::getAllOutsInRange(const ScheduleIndex start,
                                  const ScheduleIndex end) const {
  const auto f = [this](ScheduleIndex i) -> auto & {
    return getOp(scheduleToOp(i)).getOuts();
  };
  return getInRange<OpAddress>(start, end, nOps(), f);
}

std::vector<AllocWeight>
ScheduledGraph::getFwdRippleCosts(const ScheduleIndex start0,
                                  int nToShift,
                                  int firstExtCon) const {

  const int sign             = +1;
  const auto nCostsToCompute = firstExtCon - nToShift - start0;
  const auto dirOffset       = nToShift - 1;
  return getRippleCosts(start0, nToShift, sign, nCostsToCompute, dirOffset);
}

AllocWeight ScheduledGraph::getShiftCost(ScheduleIndex start0,
                                         ScheduleIndex start1,
                                         int nToShift,
                                         const Alloc &alloc) const {

  // rotate the problem so that start0 < start1
  if (start1 < start0) {
    auto old0 = start0;
    auto old1 = start1;
    start0    = old1;
    start1    = old1 + nToShift;
    nToShift  = old0 - old1;
  }

  AllocWeight fwdShiftCost = AllocWeight::negativeOne();

  // example:
  //
  // start0   = 4
  // start1   = 7
  // nToShift = 5
  //
  // 0 1 2 3 4 5 6 7 8 9 a b
  // . . . . x x x x x o o o , , , ,
  //         |         |     |
  //         x0        |     |
  //                   o0    |
  //                         o1
  const auto x0 = start0;
  const auto o0 = start0 + nToShift;
  const auto o1 = start1 + nToShift;

  // the shift:
  //                   a b c
  //                     |
  //         4           |
  // . . . . x x x x x o o o , , , ,
  //     | |  \ \ \ \ \      | |
  //     | |   \ \ \ \ \     | |
  //     | |    \ \ \ \ \    | |
  //     | |     \ \ \ \ \   | |
  //     | |      \ \ \ \ \  | |
  // . . . . o o o x x x x x , , , ,
  //           |   7
  //           |
  //         a b c (shifted back nToShift = 5)

  // how much each x is shifted forward
  auto fwdShift = start1 - start0;

  // how much each o is shifted backward
  auto bwdShift = nToShift;

  auto allocWeight  = alloc.getWeight();
  auto allocAddress = alloc.getAddress();

  // current alloc liveneness range
  auto a0 = allocToFirstSchedule(allocAddress);
  auto a1 = allocToFinalSchedule(allocAddress);

  // the different cases to consider for {a0, a1}
  //
  // . . . . x x x x o o o o , , , , ,
  //
  // {a0, a1} can be one of these 10:
  //
  //        ..   .x   .o   .,
  //             xx   xo   x,
  //                  oo   o,
  //                       ,,
  //

  //  ..         ,,          .,
  if (a1 < x0 || o1 <= a0 || (a0 < x0 && o1 <= a1)) {
    fwdShiftCost = AllocWeight(0);
  }

  //        xx                       oo
  else if ((x0 <= a0 && a1 < o0) || (o0 <= a0 && a1 < o1)) {
    fwdShiftCost = AllocWeight(0);
  }

  // .x
  else if (a0 < x0 && x0 <= a1 && a1 < o0) {
    fwdShiftCost = fwdShift * allocWeight;
  }

  // o,
  else if (o0 <= a0 && a0 < o1 && o1 <= a1) {
    fwdShiftCost = bwdShift * allocWeight;
  }

  // three remaining cases : .o  xo  x,
  else {
    const auto &indices = allocToSchedule(allocAddress);

    // for all the remaining cases, there is at least 1 post-x
    auto firstPostX =
        custom_lower_bound(indices.cbegin(), indices.cend(), o0);

    // .o
    if (a0 < x0) {
      auto lastPreO = *(firstPostX - 1);

      // .o with NONE in x
      if (lastPreO < x0) {
        fwdShiftCost = -1 * bwdShift * allocWeight;
      }
      // .o with at least one IN x
      // . . . x x x o o o , , ,
      //   |       |   |
      //   -------------
      else {
        fwdShiftCost =
            ((lastPreO - x0 + o1 - o0) - (a1 - o0 + o0 - x0)) * allocWeight;
      }
    }

    // xo   x,
    else {

      // xo
      if (a1 < o1) {
        // Examples:
        //         a0          a1
        //         x0              o1
        // . . . . x x x x x o o o , , , , ,
        //         |   |     | |
        //         -------------
        // in the above example the live period goes from 7 to 6, down by 1.
        //
        // . . . . x x x x x o o o , , , , ,
        //                 | |
        //                 ---
        // in the above example the live period goes from 2 to 8, up by 6.
        auto costBefore = a1 - a0 + 1;
        auto delta      = *firstPostX - *(firstPostX - 1);
        auto costAfter  = o1 - x0 - delta + 1;
        fwdShiftCost    = (costAfter - costBefore) * allocWeight;
      }

      // x, with at least one IN o
      else if (*firstPostX < o1) {
        // o1 <= a1
        // . . . . x x x x x o o o . . . . .
        //         |           |     |
        //         -------------------
        fwdShiftCost = ((a0 - x0) - (*firstPostX - o0)) * allocWeight;
      }

      // x, with NONE in o
      else {
        // . . . . x x x x x o o o . . . . .
        //         |     |           |
        //         -------------------
        fwdShiftCost = -1 * fwdShift * allocWeight;
      }
    }
  }
  return fwdShiftCost;
}

std::vector<OpAddress>
ScheduledGraph::getSubSchedule(const std::vector<OpAddress> &oas) const {

  // Note, we do not cache the result of this function as the graph could be
  // mutated after the first call, and keeping track of this would add a lot
  // of unnecessary complexity to the rest of the code.

  // 1. Subset `schToOp` by setting schedule index `i` to a special (otherwise
  //    impossible) value iff the op scheduled at `i` is not in the subset of
  //    ops defined by `oas`.

  const auto unassignedFlag = nOps();
  std::vector<OpAddress> subSchToOp(nOps(), unassignedFlag);

  for (const auto oa : oas) {
    // Guard invalid OpAddress.
    if (oa >= nOps()) {
      std::ostringstream oss;
      oss << "Out of range OpAddress " << oa << ". nOps = " << nOps() << ".";
      throw error(oss.str());
    }

    auto &schIdx = subSchToOp[opToSch[oa]];

    // Guard duplicates.
    if (schIdx != unassignedFlag) {
      throw error("Duplicate OpAddress " + std::to_string(oa));
    }

    schIdx = oa;
  }

  // 2. Collect only the assigned schedule indices into one vector.

  std::vector<OpAddress> subSchedule;
  subSchedule.reserve(oas.size());

  for (const auto oa : subSchToOp) {
    if (oa != unassignedFlag) {
      subSchedule.push_back(oa);
    }
  }

  return subSchedule;
}

void ScheduledGraph::applyChange(const ScheduleChange &scheduleChange,
                                 const ISummaryWriter &summaryWriter) {

  const auto nToShift = scheduleChange.getNToShift();

  const auto canonScheduleChange = scheduleChange.getCanonical();
  const auto canonStart0         = canonScheduleChange.getStart0();
  const auto canonStart1         = canonScheduleChange.getStart1();
  const auto canonNToShift       = canonScheduleChange.getNToShift();

  // see getFwdShiftCost for x/o notation
  auto x0 = canonStart0;
  auto o0 = canonStart0 + canonNToShift;
  auto o1 = canonStart1 + canonNToShift;

  auto touchedAllocs = getAllocAddresses(x0, o1);

  // An example of std::rotate
  //
  // >> std::vector<int> a(8);
  //
  // >> std::iota(a.begin(), a.end(), 0);
  // #  0 1 2 3 4 5 6 7
  // #      [     | )
  //
  // >> std::rotate(a.begin() +2, a.begin() +5, a.begin() + 6);
  // #  0 1 5 2 3 4 6 7

  updateSusceptible(x0, o0);
  updateSusceptible(o0, o1);

  // 0 schToOp
  rotate(schToOp, x0, o0, o1);

  // 1 opToSch
  for (ScheduleIndex i = x0; i < o1; ++i) {
    opToSch[scheduleToOp(i)] = i;
  }

  // 2 allocToSch
  for (auto allocAddress : touchedAllocs) {
    setAllocToSch(allocAddress);
  }

  // 3 schToAllocs
  rotate(schToAllocs, x0, o0, o1);

  const std::vector<OpAddress> consumersTouched = getAllOutsInRange(x0, o1);
  const std::vector<OpAddress> producersTouched = getAllInsInRange(x0, o1);

  // 4 opToInSch
  for (OpAddress consumerAddress : consumersTouched) {
    setOpToInSch(consumerAddress);
  }

  // 5 opToOutSch
  for (OpAddress producerAddress : producersTouched) {
    setOpToOutSch(producerAddress);
  }

  // 6 nCanFwd and nCanBwd
  updateNCanFwds(nToShift, x0, o1, producersTouched);
  updateNCanBwds(nToShift, x0, o1, consumersTouched);

  summaryWriter.appendScheduleChange(scheduleChange);
  summaryWriter.appendLivenessProfile(*this);
}

void ScheduledGraph::updateSusceptible(const ScheduleIndex a,
                                       const ScheduleIndex b) {
  if (susceptible.empty()) {
    return;
  }
  for (auto i = a; i < b; ++i) {
    const auto opAddress = scheduleToOp(i);
    for (auto inAddress : getOp(opAddress).getIns()) {
      if (opToSchedule(inAddress) < a) {
        susceptible[inAddress] = true;
        susceptible[opAddress] = true;
      }
    }
    for (auto outAddress : getOp(opAddress).getOuts()) {
      if (opToSchedule(outAddress) >= b) {
        susceptible[outAddress] = true;
        susceptible[opAddress]  = true;
      }
    }
  }
}

// TODO(T14827) : there's a faster way to do this when n2s = 1, will be
// useful when multi-threading as fast updating will become important
void ScheduledGraph::updateNCanFwds(
    const int n2s,
    const int x0,
    const int o1,
    const std::vector<OpAddress> &producersTouched) {

  // for all x in ScheduleIndices, update how far forward the range [x, x+
  // n2s) can shift forwards

  // determine which schedule indices may have changed
  ScheduleIndex startCanFwdRange = x0;
  ScheduleIndex endCanFwdRange   = o1;
  for (OpAddress p : producersTouched) {
    startCanFwdRange = std::min(startCanFwdRange, opToSch[p]);
  }
  startCanFwdRange -= (n2s + 1);
  startCanFwdRange = std::max(0, startCanFwdRange);
  endCanFwdRange   = std::min(endCanFwdRange, nOps_i32() - n2s + 1);

  // reset the schedule indices which may have changed
  for (ScheduleIndex i = startCanFwdRange; i < endCanFwdRange; ++i) {
    auto i_u64     = static_cast<uint64_t>(i);
    nCanFwd[i_u64] = getFirstConsumer(i, n2s) - i - n2s;
  }
}

// TODO(T14827) : there's a faster way to do this when n2s = 1.
void ScheduledGraph::updateNCanBwds(
    int n2s,
    int x0,
    int o1,
    const std::vector<OpAddress> &consumersTouched) {
  ScheduleIndex startCanBwdRange = x0;
  ScheduleIndex endCanBwdRange   = o1;
  for (OpAddress c : consumersTouched) {
    endCanBwdRange = std::max(endCanBwdRange, opToSch[c] + 1);
  }
  startCanBwdRange -= n2s + 1;
  startCanBwdRange = std::max(0, startCanBwdRange);
  endCanBwdRange   = std::min(endCanBwdRange, nOps_i32() - n2s + 1);
  for (ScheduleIndex i = startCanBwdRange; i < endCanBwdRange; ++i) {
    auto i_u64     = static_cast<uint64_t>(i);
    nCanBwd[i_u64] = i - getLastProducer(i, n2s) - 1;
  }
}

std::vector<AllocWeight> ScheduledGraph::getSchToLiveness() const {
  std::vector<AllocWeight> s2l;
  s2l.reserve(nOps());
  s2l.clear();
  auto deltaLiveness = getDeltaLiveness();
  s2l.push_back(deltaLiveness[0]);
  for (uint64_t i = 0; i < nOps(); ++i) {
    s2l.push_back(s2l.back() + deltaLiveness[i + 1]);
  }
  return s2l;
}

void ScheduledGraph::setSchToLiveness() {
  schToLiveness = getSchToLiveness();
}

ScheduleIndex ScheduledGraph::getFirstConsumer(ScheduleIndex start,
                                               const int nToShift) const {

  //                       ......xxxxxxxxxx...............
  // all consumers                |    | |       | |  |
  //                                             |
  //                                       first consumer

  // if there is no consumer, this is what is returned:
  ScheduleIndex upper = nOps_i32();

  for (ScheduleIndex i = start; i < start + nToShift; ++i) {
    OpAddress opAddress = scheduleToOp(i);
    auto firstFromEnd =
        custom_lower_bound(opToOutSchedule(opAddress).cbegin(),
                           opToOutSchedule(opAddress).cend(),
                           start + nToShift);
    if (firstFromEnd != opToOutSchedule(opAddress).cend()) {
      upper = std::min(upper, *firstFromEnd);
    }
  }

  return upper;
}

ShiftAndCost
ScheduledGraph::getBestShiftRippleAlgo(const ScheduleIndex start,
                                       const int nToShift) const {

  ScheduleIndex bestShift{0};
  AllocWeight bestCost{0};

  // see comment-I for how this bound works
  if (getNCanBwd(start) >= nToShift) {
    ScheduleIndex lastProducer = start - getNCanBwd(start) - 1;
    auto bwdCosts = getBwdRippleCosts(start, nToShift, lastProducer);
    for (ScheduleIndex proposedStart = lastProducer + 1;
         proposedStart < start;
         ++proposedStart) {
      auto bwdCostIndex = static_cast<uint64_t>(start - 1 - proposedStart);
      auto cost         = bwdCosts[bwdCostIndex];
      if (cost < bestCost && isLinkPreserving(proposedStart,
                                              proposedStart + nToShift,
                                              start - proposedStart)) {

        bestCost  = cost;
        bestShift = proposedStart - start;
      }
    }
  }

  // see comment-I for how this bound works
  if (getNCanFwd(start) >= nToShift) {
    // [start + 1, firstConsumer - nToShift]
    ScheduleIndex firstConsumer = start + getNCanFwd(start) + nToShift;
    auto fwdCosts = getFwdRippleCosts(start, nToShift, firstConsumer);
    for (uint64_t i = 0; i < fwdCosts.size(); ++i) {
      int shift = static_cast<int>(i) + 1;
      if (fwdCosts[i] < bestCost &&
          isLinkPreserving(start, start + shift, nToShift)) {
        bestShift = shift;
        bestCost  = fwdCosts[i];
      }
    }
  }

  ShiftAndCost best{bestShift, bestCost};
  return best;
}

std::string ScheduledGraph::getLivenessString() const {

  std::vector<std::string> sIndex;
  std::vector<std::string> sIns;
  std::vector<std::string> sLinkTo;
  std::vector<std::string> sOuts;
  std::vector<std::string> sAllocs;
  std::vector<std::string> sLiveness;
  std::vector<std::string> sName;

  for (uint64_t i = 0; i < nOps(); ++i) {

    auto address   = schToOp[i];
    const auto &op = getOp(address);

    std::ostringstream ossIns;
    ossIns << opToInSch[address];
    std::ostringstream ossLinkTo;
    ossLinkTo << (op.hasForwardLink() ? '+' : ' ');
    std::ostringstream ossName;
    ossName << getOp(address).getDebugString();
    std::ostringstream ossOuts;
    ossOuts << opToOutSch[address];
    std::ostringstream ossAllocs;
    ossAllocs << schToAllocs[i];

    sIndex.push_back(std::to_string(i));
    sLiveness.push_back(toString(schToLiveness[i]));
    sIns.push_back(ossIns.str());
    sLinkTo.push_back(ossLinkTo.str());
    sOuts.push_back(ossOuts.str());
    sAllocs.push_back(ossAllocs.str());
    sName.push_back(ossName.str());
  }

  std::vector<util::StringColumn> stringCols{{"Index", sIndex},
                                             {"Name", sName},
                                             {"Ins", sIns},
                                             {"LinkTo", sLinkTo},
                                             {"Outs", sOuts},
                                             {"Allocs", sAllocs},
                                             {"Liveness", sLiveness}};

  return alignedColumns(stringCols);
}

ScheduleIndex ScheduledGraph::getLastProducer(const ScheduleIndex start,
                                              const int nToShift) const {

  // Consider a range [start, start + nToShift)  of Ops in a schedule, denoted
  // by x's below:
  //
  //            start
  //            |     start + nToShift
  //            |     |
  //  ..........xxxxxx..........
  //  0123456789 etc               <-- schedule index.
  //
  // Each of the x's has a (possibly empty) set of input (a.k.a. producer)
  // Ops. Consider the union of the schedule indices of all of these
  // producers. For example,
  //
  //                          0         10        20
  //                          01234567890123456789012345  <-- schedule index.
  //                          .............xxxxxxxxxxx.....
  // all producers of all x's:  |  | |   |   ||  | |
  //                                     |
  //                             (last producer)
  //
  // The set of indices of producers of all x's in the above example is
  // {2,4,7,11,15,16,19,21}.
  //
  // This function returns the largest schedule index in this set which is
  // less than "start", which in this example is 11.
  //
  // If the set is empty, then -1 is returned.
  //

  // if there is no producer less than start for any x, this is what is
  // returned:
  ScheduleIndex lower = -1;

  for (ScheduleIndex i = start; i < start + nToShift; ++i) {
    OpAddress opAddress = scheduleToOp(i);

    auto firstFromStart =
        custom_lower_bound(opToInSchedule(opAddress).cbegin(),
                           opToInSchedule(opAddress).cend(),
                           start);

    // It is possible that there were no consumers preceding start, in the
    // case where there was:
    if (firstFromStart != opToInSchedule(opAddress).cbegin()) {
      lower = std::max(lower, *(std::prev(firstFromStart)));
    }
  }

  return lower;
}

bool ScheduledGraph::isLinkPreserving(const ScheduleIndex start0,
                                      const ScheduleIndex start1,
                                      const int nToShift) const {

  const auto x0 = start0;
  const auto o0 = start0 + nToShift;
  const auto o1 = start1 + nToShift;

  const auto &xStartOp = getOp(scheduleToOp(x0));
  const auto &xEndOp   = getOp(scheduleToOp(o0 - 1));
  const auto &oStartOp = getOp(scheduleToOp(o0));
  const auto &oEndOp   = getOp(scheduleToOp(o1 - 1));

  return !(xStartOp.hasBackwardLink() || xEndOp.hasForwardLink() ||
           oStartOp.hasBackwardLink() || oEndOp.hasForwardLink());
}

void ScheduledGraph::confirmShiftAndCost(const ScheduleIndex start0,
                                         const int nToShift,
                                         const ShiftAndCost &shiftAndCost,
                                         const RotationAlgo algo) const {

  auto debugShiftAndCost = getBestShiftSimpleAlgo(start0, nToShift);
  if (debugShiftAndCost != shiftAndCost) {
    const std::string runAlgoStr = getRotationAlgoString(algo);
    const std::string basicAlgoStr =
        getRotationAlgoString(RotationAlgo::SIMPLE);
    std::ostringstream oss;
    oss << "\nDifferent results between " << runAlgoStr << " and "
        << basicAlgoStr << " in debug for greedyRotate. "
        << "The current state of schedule is \n"
        << getLivenessString() << "With " << basicAlgoStr
        << ", the suggestion is\n  " << debugShiftAndCost << "\nand with"
        << runAlgoStr << ", the suggestion is\n  " << shiftAndCost
        << ".\nThis for start0 = " << start0
        << " and nToShift = " << nToShift;
    throw error(oss.str());
  }
}

ScheduledGraph::ScheduledGraph(Graph &&gInitial,
                               const Settings &settings,
                               const ISummaryWriter &summaryWriter)
    : swatch_(std::string("ScheduledGraphTimeLogger")) {

  const auto stopwatch =
      timeLogger().scopedStopwatch("ScheduledGraph::ScheduledGraph");

  const auto mightRequireInitialGraph = summaryWriter.mightWrite(gInitial);
  if (mightRequireInitialGraph) {
    graph = gInitial;
  } else {
    graph = std::move(gInitial);
  }

  initialize(settings.kahnDecider(),
             settings.seed(),
             settings.tcos(),
             summaryWriter);
  greedyRotate(settings.rotationAlgo(),
               settings.debugMode(),
               settings.seed(),
               settings.rotationTermination(),
               summaryWriter);

  constexpr double thresholdPercentage{0.0};

  std::ostringstream oss;
  oss << "Breakdown of the time spent scheduling the provided graph:\n"
      << timeLogger().str(thresholdPercentage);

  const auto summaryString = oss.str();
  log().info(summaryString);

  if (mightRequireInitialGraph &&
      summaryWriter.willWrite(gInitial,
                              getTimeLogger().sinceConstruction())) {
    summaryWriter.write(gInitial,
                        getGraph(),
                        getTimeLogger().sinceConstruction(),
                        summaryString);
  }

  summaryWriter.writeFinalSchedule(schToOp);
}

ScheduledGraph::ScheduledGraph(Graph &&g,
                               const std::map<std::string, std::string> &m)
    : ScheduledGraph(std::move(g), Settings(m), FileWriter::None()) {}

ScheduledGraph::ScheduledGraph(Graph &&g,
                               const KahnDecider &kd,
                               const TransitiveClosureOptimizations tco,
                               const RotationTermination rt,
                               const RotationAlgo algo,
                               const uint32_t seed,
                               const ISummaryWriter &summaryWriter_,
                               const DebugMode debugMode)
    : ScheduledGraph(std::move(g),
                     Settings(kd, tco, rt, algo, seed, debugMode),
                     summaryWriter_) {}

void ScheduledGraph::greedyRotate(RotationAlgo algo,
                                  DebugMode debugMode,
                                  uint32_t seed,
                                  RotationTermination rt,
                                  const ISummaryWriter &summaryWriter) {

  const auto stopwatch = timeLogger().scopedStopwatch("greedyRotate");

  summaryWriter.appendLivenessProfile(*this);

  if (log().shouldLog(logging::Level::Debug)) {
    std::ostringstream oss0;
    oss0 << '\n'
         << spaces << "debug=" << debugMode << '\n'
         << spaces << "seed=" << seed << '\n'
         << spaces << "timeLimitSeconds=" << rt.maxSeconds() << '\n'
         << spaces << "swapLimitCount=" << rt.maxRotations();
    log().debug(oss0.str());
  }

  std::mt19937 g(seed);

  auto resetSusceptibleTrue = [this]() {
    susceptible.resize(nOps());
    std::fill(susceptible.begin(), susceptible.end(), true);
  };

  auto resetSusceptibleFalse = [this]() {
    susceptible.resize(nOps());
    std::fill(susceptible.begin(), susceptible.end(), false);
  };

  // look for moves of this shift length
  int nToShift{1};
  bool continueShifting =
      (rt.maxSeconds() <= 0 || rt.maxRotations() <= 0) ? false : true;

  // at a given shift, there may be multiple rounds
  int nChangesInCurrentRound{0};
  AllocWeight deltaWeightCurrentRound{0};

  int64_t nChangesInTotal{0};
  int64_t nResetsToOne{0};
  int64_t nShiftingRounds{0};

  // one "shift" phase can consist of multple "rounds"
  double timeSpentInCurrentRound{0.0};
  double timeSpentInTotal{0.0};

  // there have been no-recorded improvements since last at nToShift = 1
  bool noChangeSinceStart{true};

  std::vector<OpAddress> allOpAddresses(nOps());

  std::iota(allOpAddresses.begin(), allOpAddresses.end(), 0UL);
  // Randomize the order in which indices are processed:
  std::shuffle(allOpAddresses.begin(), allOpAddresses.end(), g);

  auto startCurrentShift = std::chrono::high_resolution_clock::now();

  // used in a check for the correctness of all computed improvements
  const AllocWeight initSumLiveness = getSumLiveness();
  const AllocWeight initMaxLiveness = getMaxLiveness();
  AllocWeight totalDeltaSumLiveness{0};

  resetSusceptibleTrue();

  while (continueShifting) {

    ++nShiftingRounds;

    auto susceptibleCurrent = susceptible;
    resetSusceptibleFalse();

    auto startCurrentRound = std::chrono::high_resolution_clock::now();

    nChangesInCurrentRound  = 0;
    deltaWeightCurrentRound = AllocWeight::zero();
    for (auto opAddress0 : allOpAddresses) {

      auto start0     = opToSchedule(opAddress0);
      const auto &op0 = getOp(opAddress0);
      if (start0 > nOps_i32() - nToShift) {
        continue;
      }

      const auto &op1 = getOp(scheduleToOp(start0 + nToShift - 1));

      // if links at end or start, can igonore. Consider
      //
      //    a op0 b c op1 d
      //      -----------
      //
      // if a is linked to op0, any shift of op0-b-c-op1 would break this
      // link: not allowed
      //
      // if op1 is linked to d, any shift of op0-b-c-op1 would break this
      // link: not allowed.
      //
      if (op0.hasBackwardLink() || op1.hasForwardLink()) {
        continue;
      }

      if (std::all_of(schToOp.begin() + start0,
                      schToOp.begin() + start0 + nToShift,
                      [&susceptibleCurrent](OpAddress a) {
                        return !susceptibleCurrent[a];
                      })) {
        continue;
      }

      ShiftAndCost shiftAndCost{-1, -1 * AllocWeight::negativeOne()};
      if (algo == RotationAlgo::RIPPLE) {
        shiftAndCost = getBestShiftRippleAlgo(start0, nToShift);
      } else {
        shiftAndCost = getBestShiftSimpleAlgo(start0, nToShift);
      }

      if (debugMode == DebugMode::On) {
        confirmShiftAndCost(start0, nToShift, shiftAndCost, algo);
      }
      if (shiftAndCost.getCost() < AllocWeight(0)) {
        auto start1 = start0 + shiftAndCost.getShift();
        ScheduleChange scheduleChange{start0, start1, nToShift};

        applyChange(scheduleChange, summaryWriter);

        if (debugMode == DebugMode::On) {
          assertCorrectness();
        }
        ++nChangesInCurrentRound;
        deltaWeightCurrentRound += shiftAndCost.getCost();
        totalDeltaSumLiveness += shiftAndCost.getCost();
      }
    }

    nChangesInTotal += nChangesInCurrentRound;
    noChangeSinceStart = noChangeSinceStart && nChangesInCurrentRound == 0;

    auto finishCurrentRound = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedCurrentRound =
        finishCurrentRound - startCurrentRound;
    timeSpentInCurrentRound = elapsedCurrentRound.count();
    timeSpentInTotal += timeSpentInCurrentRound;
    if (timeSpentInTotal > rt.maxSeconds()) {
      continueShifting = false;
    }
    if (nChangesInTotal >= rt.maxRotations()) {
      continueShifting = false;
    }

    auto oldNToShift = nToShift;

    std::ostringstream oss;
    oss << "In round #" << nShiftingRounds << ", " << nChangesInCurrentRound
        << " changes. ";

    if (noChangeSinceStart) {
      oss << "No changes since start, climbing " << nToShift << " --> "
          << nToShift + 1;
      ++nToShift;
      resetSusceptibleTrue();
    } else if (nChangesInCurrentRound == 0) {
      oss << "No changes in round, descending " << nToShift
          << " -->  1. Descent #" << nResetsToOne << '.';
      nToShift           = 1;
      noChangeSinceStart = true;
      ++nResetsToOne;
      resetSusceptibleTrue();
    } else {
      oss << "staying at " << nToShift;
      nToShift = oldNToShift;
    }

    log().info(oss.str());

    if (oldNToShift != nToShift) {
      updateCanCan(oldNToShift, nToShift);
      auto finishCurrentShift = std::chrono::high_resolution_clock::now();
      startCurrentShift       = finishCurrentShift;
    }

    if (noChangeSinceStart) {
      auto nToConsider = 0;
      for (uint64_t i = 0; i < nCanFwd.size(); ++i) {
        if (nCanFwd[i] > nToShift || nCanBwd[i] > nToShift) {
          ++nToConsider;
        }
      }
      if (nToConsider == 0) {
        continueShifting = false;
      }
    }
  }

  // Algorithm complete. Gather final statistics and test for error.

  setSchToLiveness();

  auto finalMaxLiveness = getMaxLiveness();
  auto finalSumLiveness = getSumLiveness();

  auto absErr =
      absolute(finalSumLiveness - initSumLiveness - totalDeltaSumLiveness);
  auto relErr = absErr / (1.0 + absolute(totalDeltaSumLiveness));
  for (auto x : relErr.get()) {
    if (x > 1e-5) {
      std::ostringstream oss2;
      oss2 << "An error might have occurred in greedyRotate. "
           << "The running accumulation of calculated improvements is "
           << totalDeltaSumLiveness << '.' << ' '
           << "The difference between initial and final liveness sums is "
           << finalSumLiveness - initSumLiveness << '.';
      throw error(oss2.str());
    }
  }

  if (log().shouldLog(logging::Level::Debug)) {
    std::ostringstream oss0;
    oss0 << '\n'
         << spaces << "init sum liveness =  " << initSumLiveness << '\n'
         << spaces << "final sum liveness = " << finalSumLiveness << '.'
         << '\n'
         << spaces << "init max liveness =  " << initMaxLiveness << '\n'
         << spaces << "final max liveness = " << finalMaxLiveness << '.';
    log().info(oss0.str());
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
