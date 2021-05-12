// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>

#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/filteredschedule.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/shift/schedulechange.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/solutioncache.hpp>
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

void updateFromFirst(AllocWeight &lwr,
                     AllocWeight &upp,
                     const AllocWeight &w,
                     const transitiveclosure::IsFirst isFirst) {
  switch (isFirst) {
  // If an Op is definitely not the first consumer of an allocation, the
  // allocation definitely does not increase liveness
  case (transitiveclosure::IsFirst::No): {
    break;
  }
  case (transitiveclosure::IsFirst::Maybe): {
    // If an Op might be the first consumer of an allocation, the allocation
    // might increase liveness. The upper-bound on liveness is therefore
    // increased
    upp += w;
    break;
  }
  case (transitiveclosure::IsFirst::Yes): {
    lwr += w;
    upp += w;
    break;
  }
  }
}

void updateFromFinal(AllocWeight &lwr,
                     AllocWeight &upp,
                     const AllocWeight &w,
                     const transitiveclosure::IsFinal isFinal) {
  switch (isFinal) {
  case (transitiveclosure::IsFinal::No): {
    break;
  }
  case (transitiveclosure::IsFinal::Maybe): {
    lwr -= w;
    break;
  }
  case (transitiveclosure::IsFinal::Yes): {
    lwr -= w;
    upp -= w;
    break;
  }
  }
}

void updateFromFirstFinal(AllocWeight &lwr,
                          AllocWeight &upp,
                          const AllocWeight &w,
                          const std::tuple<transitiveclosure::IsFirst,
                                           transitiveclosure::IsFinal> ff) {
  updateFromFirst(lwr, upp, w, std::get<0>(ff));
  updateFromFinal(lwr, upp, w, std::get<1>(ff));
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

void ScheduledGraph::initialize(const KahnTieBreaker kahnTie,
                                const uint32_t seed,
                                const TransitiveClosureOptimizations tco) {

  std::ostringstream oss;
  oss << "Graph::initialize() entered for Graph with " << nOps() << " Ops, "
      << graph.nAllocs() << " Allocs, " << graph.nConstraints()
      << " constraints. ";
  log().info(oss.str());

  applyTransitiveClosureOptimizations(tco);

  //
  // schToOp. Vanilla run of Kahn's O(E) algorithm, random tie-breaks
  schToOp = kahn(graph, kahnTie, seed);

  //
  // opToSch
  opToSch.reserve(nOps());
  opToSch.clear();

  for (ScheduleIndex i = 0; i < nOps_i32(); ++i) {
    opToSch[scheduleToOp(i)] = i;
  }

  //
  // allocToSch
  allocToSch.resize(graph.nAllocs());
  allocToSch.clear();

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

bool ScheduledGraph::constrainParallelChains() {
  std::vector<std::array<OpAddress, 2>> newConstraints;
  for (OpAddress a = 0; a < nOps(); ++a) {
    auto identicalIns = graph.getIdenticalIns(a);
    if (identicalIns.size() <= 1) {
      continue;
    }
    const auto aChain = graph.tightChainFrom(a);
    const auto aEnd   = aChain.back();
    for (auto b : identicalIns) {
      if (b == a) {
        continue;
      }
      auto bChain       = graph.tightChainFrom(b);
      auto bEnd         = bChain.back();
      const auto &aOuts = getOp(aEnd).getOuts();
      const auto &bOuts = getOp(bEnd).getOuts();
      if (!(aOuts == bOuts && (aChain.size() >= bChain.size()))) {
        continue;
      }

      auto chainLength = bChain.size();
      bool canInsertConstraints{true};

      auto runningUpp = AllocWeight::zero();
      auto runningLow = AllocWeight::zero();

      for (uint64_t i = 0; i < chainLength; ++i) {

        auto uppA = upperBoundChange[aChain[i]];
        auto lowB = lowerBoundChange[bChain[i]];

        for (auto allocAddress : getOp(bChain[i]).getAllocs()) {

          // Determine if a shared alloc can be removed:
          bool canRemove = true;
          for (uint64_t j = 0; j < chainLength; ++j) {
            if (getOp(aChain[i]).hasAlloc(allocAddress) !=
                getOp(bChain[i]).hasAlloc(allocAddress)) {
              canRemove = false;
            }
          }
          if (!canRemove) {
            continue;
          }

          // Remove shared: alloc contribution
          const auto &alloc = graph.getAlloc(allocAddress);
          const auto &all   = alloc.getOps();
          auto negW         = -1 * alloc.getWeight();

          AllocWeight dummy = AllocWeight::zero();
          {

            const auto relPoss =
                transitiveClosure.getExtremumStatus(aChain[i], all);
            updateFromFirstFinal(dummy, uppA, negW, relPoss);
          }

          {

            const auto relPoss =
                transitiveClosure.getExtremumStatus(bChain[i], all);
            updateFromFirstFinal(lowB, dummy, negW, relPoss);
          }
        }

        runningUpp += uppA;
        runningLow += lowB;

        if (runningUpp < runningLow ||
            (runningUpp == runningLow && aChain[i] < bChain[i])) {
        } else {
          canInsertConstraints = false;
          break;
        }
      }

      if (canInsertConstraints) {
        for (uint64_t i = 0; i < chainLength; ++i) {
          if (!getOp(aChain[i]).hasOut(bChain[i])) {
            newConstraints.push_back({aChain[i], bChain[i]});
          }
        }
      }
    }
  }
  for (auto constraint : newConstraints) {
    auto from = std::get<0>(constraint);
    auto to   = std::get<1>(constraint);
    graph.insertConstraint(from, to);
  }

  log().debug(
      std::to_string(newConstraints.size()) +
      " new constraints inserted in graph::constrainParallelChains()");

  return !newConstraints.empty();
}

bool ScheduledGraph::slideLinks() {
  bool wasChange{false};
  auto linkChains = graph.getLinkChains();
  for (const auto &chain : linkChains) {
    for (uint64_t i = 0; i < chain.size(); ++i) {
      auto id = chain[i];

      if (i != chain.size() - 1) {
        const auto outs = getOp(id).getOuts();
        for (const auto outId : outs) {
          if (getOp(id).getForwardLink() != outId) {
            graph.removeConstraint(id, outId);
            graph.insertConstraint(chain.back(), outId);
            wasChange |= true;
          }
        }
      }
      if (i != 0) {
        const auto ins = getOp(id).getIns();
        for (const auto inId : ins) {
          if (getOp(id).getBackwardLink() != inId) {
            graph.removeConstraint(inId, id);
            graph.insertConstraint(inId, chain[0]);
            wasChange |= true;
          }
        }
      }
    }
  }

  return wasChange;
}

void ScheduledGraph::updateTransitiveClosure(
    const std::vector<std::vector<OpAddress>> &edges) {
  if (log().shouldLog(logging::Level::Debug)) {
    std::ostringstream oss;
    oss << "Updating TransitiveClosure with "
        << std::accumulate(
               edges.cbegin(),
               edges.cend(),
               0,
               [](size_t a, const auto &x) { return a + x.size(); })
        << " new constraints. ";
    log().debug(oss.str());
  }

  transitiveClosure.update(edges);
  finalizeTransitiveClosure();
}

void ScheduledGraph::applyTransitiveClosureOptimizations(
    const TransitiveClosureOptimizations &tco) {

  if (tco.allOptimizationsOff()) {
    return;
  }

  bool wasChange{true};
  int iteration{0};
  const auto iterStr = "iteration = " + std::to_string(iteration);

  std::vector<std::vector<OpAddress>> prevGraphEdges;
  while (wasChange && iteration < tco.maxIterations()) {

    if (iteration == 0) {
      log().debug("Initializing TransitiveClosure," + iterStr);
      initializeTransitiveClosure();
    } else {
      const auto dff = graph.constraintDiff(prevGraphEdges);
      // As Updating a TransitiveClosure takes significantly more time for a
      // large number of edges, we prefer to re-initialize when the number of
      // edges is "large";
      const int nLarge = nOps_i32() / 10;
      if (std::accumulate(
              dff.cbegin(), dff.cend(), 0, [](size_t x, const auto &y) {
                return x + y.size();
              }) < nLarge) {

        log().debug("Updating TransitiveClosure, " + iterStr);
        updateTransitiveClosure(dff);
      } else {
        log().debug("Re-initializing TransitiveClosure,  " + iterStr);
        initializeTransitiveClosure();
      }
    }

    log().debug("Storing Graph edges, to detect changes in next iteration");
    prevGraphEdges = graph.getForwardEdges();

    log().debug("Applying TCO slideLinks");
    wasChange = slideLinks();

    if (tco.constrainWeightSeparatedGroups()) {
      log().debug("Applying TCO constrainWeightSeparatedGroups.");
      wasChange |= constrainWeightSeparatedGroups();
    }

    if (tco.constrainParallelChains()) {
      log().debug("Applying TCO constrainParallelChains.");
      wasChange |= constrainParallelChains();
    }

    if (tco.linkTightDrops()) {
      log().debug("Applying TCO linkTightDrops.");
      wasChange |= linkTightDrops();
    }

    if (tco.linkCloseTightPairs()) {
      log().debug("Applying TCO linkCloseTightPairs.");
      wasChange |= linkCloseTightPairs();
    }
    ++iteration;
  }
}

void ScheduledGraph::processWeightSeparatedIdenticalIns(
    const std::vector<OpAddress> &identicalIns,
    std::vector<std::array<OpAddress, 2>> &newConstraints) const {

  // for (a,b) can we insert a'->b for any a' which are post a?
  for (auto a : identicalIns) {
    for (auto b : identicalIns) {
      if (upperBoundChange[a] <= lowerBoundChange[b] && a != b) {

        // Here we do a depth first search, starting at b, stopping when we
        // reach an Op with is unconstrained with respect to t a.
        //
        // The Ops found end up in this vector:
        std::vector<OpAddress> postBs;
        std::vector<OpAddress> toProcess{b};
        std::vector<OpAddress> seen{b};
        while (!toProcess.empty()) {
          const auto nxt = toProcess.back();
          toProcess.pop_back();
          if (!transitiveClosure.constrained(a, nxt)) {
            postBs.push_back(nxt);
            for (auto out : getOp(nxt).getOuts()) {
              if (std::find(seen.cbegin(), seen.cend(), out) == seen.cend()) {
                seen.push_back(out);
                toProcess.push_back(out);
              }
            }
          }
        }

        auto lb = lowerBoundChange[b];
        for (auto postB : postBs) {
          lb = std::min(lb, lowerBoundChange[postB]);
        }

        if (upperBoundChange[a] <= lb) {

          auto nPostBoth  = transitiveClosure.nPostPost(a, b);
          auto candidates = getFilteredSchedule(
              graph, a, [this, lb, b, nPostBoth](OpAddress x) {
                return upperBoundChange[x] <= lb &&
                       (transitiveClosure.nPostPost(b, x) == nPostBoth);
              });

          if (a < b || std::any_of(candidates.cbegin(),
                                   candidates.cend(),
                                   [lb, this](OpAddress postA) {
                                     return upperBoundChange[postA] < lb;
                                   })) {
            for (auto aPrime : candidates) {
              newConstraints.push_back({aPrime, b});
            }
          }
        }
      }
    }
  }
}

bool ScheduledGraph::constrainWeightSeparatedGroups() {

  std::vector<bool> processed(nOps(), false);

  std::vector<std::array<OpAddress, 2>> newConstraints;
  for (OpAddress add0 = 0; add0 < nOps(); ++add0) {
    if (processed[add0]) {
      continue;
    }
    auto identicalIns = graph.getIdenticalIns(add0);
    for (auto id0 : identicalIns) {
      processed[id0] = true;
    }

    if (identicalIns.size() < 2) {
      continue;
    }

    processWeightSeparatedIdenticalIns(identicalIns, newConstraints);
  }

  for (auto constraint : newConstraints) {
    auto from = std::get<0>(constraint);
    auto to   = std::get<1>(constraint);
    graph.insertConstraint(from, to);
  }

  log().debug(
      std::to_string(newConstraints.size()) +
      " new constraints inserted in graph::constrainWeightSeparatedGroups()");

  return !newConstraints.empty();
}

void ScheduledGraph::finalizeTransitiveClosure() {

  const auto fwdEdges = graph.getForwardEdges();

  const auto redundants = transitiveClosure.getFlattenedRedundants(fwdEdges);
  log().debug("Removing " + std::to_string(redundants.size()) +
              " redundant TransitiveClosure edges/constraints.");
  for (const auto x : redundants) {
    graph.removeConstraint(std::get<0>(x), std::get<1>(x));
  }

  auto zero        = AllocWeight::zero();
  lowerBoundChange = std::vector<AllocWeight>(nOps(), zero);
  upperBoundChange = std::vector<AllocWeight>(nOps(), zero);

  // initializing lowerBoundChange and upperBoundChange
  log().debug("Initializing lowerBoundChange and upperBoundChange.");
  for (const auto &alloc : graph.getAllocs()) {
    auto relativePositions =
        transitiveClosure.getExtremumStatuses(alloc.getOps());

    // Logic check:
    if (relativePositions.size() != alloc.getOps().size()) {
      std::ostringstream oss;
      oss << "There were " << alloc.getOps().size()
          << " passed into the function getExtremumStatuses, but "
          << relativePositions.size()
          << " values were returned. There should be value entry returned "
          << "for every Op. ";
      throw error(oss.str());
    }

    for (uint64_t opIndex = 0; opIndex < alloc.nOps(); ++opIndex) {
      auto opId = alloc.getOps()[opIndex];
      updateFromFirstFinal(lowerBoundChange[opId],
                           upperBoundChange[opId],
                           alloc.getWeight(),
                           relativePositions[opIndex]);
    }
  }
}

void ScheduledGraph::initializeTransitiveClosure() {
  transitiveClosure =
      transitiveclosure::TransitiveClosure(graph.getForwardEdges());
  finalizeTransitiveClosure();
}

bool ScheduledGraph::linkTightDrops() {
  std::vector<std::array<OpAddress, 2>> newLinks;
  for (const auto tightPair : graph.getTightPairs()) {
    OpAddress before = std::get<0>(tightPair);
    OpAddress after  = std::get<1>(tightPair);
    if (upperBoundChange[after] <= lowerBoundChange[before]) {
      if (!getOp(before).hasForwardLink() &&
          !getOp(after).hasBackwardLink()) {
        newLinks.push_back(tightPair);
      }
    }
  }
  for (auto link : newLinks) {
    graph.insertLink(std::get<0>(link), std::get<1>(link));
  }
  log().debug(std::to_string(newLinks.size()) +
              " new links inserted in Graph::linkTightDrops()");
  return !newLinks.empty();
}

bool ScheduledGraph::linkCloseTightPairs() {
  std::vector<std::array<OpAddress, 2>> newLinks;

  for (const auto tightPair : graph.getTightPairs()) {
    auto before = std::get<0>(tightPair);
    auto after  = std::get<1>(tightPair);
    if (getOp(before).hasForwardLink()) {
      continue;
    }

    auto L = std::min(lowerBoundChange[before], lowerBoundChange[after]);
    auto U = std::max(upperBoundChange[before], upperBoundChange[after]);

    auto getCanTie = [this, L, U](OpAddress opId) {
      using namespace transitiveclosure;
      for (uint64_t i = 0; i < transitiveClosure.getNBitSetsPerOp(); ++i) {
        auto index     = opId * transitiveClosure.getNBitSetsPerOp() + i;
        BitSet neither = transitiveClosure.getFwdEdgeSet()[index] |
                         transitiveClosure.getBwdEdgeSet()[index];
        neither.flip();
        if (neither.any()) {
          for (uint64_t shift = 0; shift < BitSetSize; ++shift) {
            auto id = i * BitSetSize + shift;

            if (id != opId && id < nOps() && neither[shift]) {

              //      L     U
              //  ....xxxxxxx..  -- a
              //  ..xxxxx......  -- b
              //    l   u
              //  ==> intersection if L < u && l < U
              const auto u = upperBoundChange[id];
              const auto l = lowerBoundChange[id];

              if (L < u && l < U) {
                return false;
              }
            }
          }
        }
      }
      return true;
    };

    bool canTie = getCanTie(before);

    if (canTie) {
      if (!getOp(before).hasForwardLink()) {
        newLinks.push_back(tightPair);
      }
    }
  }

  for (auto link : newLinks) {
    graph.insertLink(std::get<0>(link), std::get<1>(link));
  }
  log().debug(std::to_string(newLinks.size()) +
              " new links inserted in Graph::linkCloseTightPairs()");
  return !newLinks.empty();
}

std::vector<OpAddress>
ScheduledGraph::linklessKahn(const Graph &g,
                             const KahnTieBreaker kahnTie,
                             const uint32_t seed) {

  std::vector<OpAddress> sch;

  sch.reserve(g.nOps());
  sch.clear();

  std::vector<OpAddress> outstanding;
  outstanding.reserve(g.nOps());
  std::vector<OpAddress> ready;
  for (OpAddress i = 0; i < g.nOps(); ++i) {
    outstanding.push_back(g.getOp(i).nIns());
    if (outstanding[i] == 0) {
      ready.push_back(i);
    }
  }

  std::mt19937 gen(seed);

  if (kahnTie == KahnTieBreaker::RANDOM) {
    while (!ready.empty()) {
      std::shuffle(ready.begin(), ready.end(), gen);
      OpAddress address = ready.back();
      sch.push_back(address);
      ready.pop_back();
      for (auto cAddress : g.getOp(address).getOuts()) {
        --outstanding[cAddress];
        if (outstanding[cAddress] == 0) {
          ready.push_back(cAddress);
        }
      }
    }
  }

  else if (kahnTie == KahnTieBreaker::FIFO) {
    while (!ready.empty()) {
      OpAddress address = ready.back();
      sch.push_back(address);
      ready.pop_back();
      for (auto cAddress : g.getOp(address).getOuts()) {
        --outstanding[cAddress];
        if (outstanding[cAddress] == 0) {
          ready.push_back(cAddress);
        }
      }
    }
  }

  // GREEDY
  else if (kahnTie == KahnTieBreaker::GREEDY) {

    std::vector<int> nOutstandingForAlloc(g.nAllocs());
    std::vector<bool> allocLive(g.nAllocs(), false);
    for (const auto &alloc : g.getAllocs()) {
      nOutstandingForAlloc[alloc.getAddress()] = alloc.nOps_i32();
    }

    auto deltaLive = [&g, &nOutstandingForAlloc, &allocLive](OpAddress a) {
      AllocWeight delta{0};
      for (auto allocAddress : g.getOp(a).getAllocs()) {
        const auto allocWeight = g.getAlloc(allocAddress).getWeight();
        if (nOutstandingForAlloc[allocAddress] == 1) {
          delta -= allocWeight;
        }
        if (!allocLive[allocAddress]) {
          delta += allocWeight;
        }
      }
      return delta;
    };

    while (!ready.empty()) {
      std::shuffle(ready.begin(), ready.end(), gen);

      auto bestIter         = ready.cbegin();
      AllocWeight bestDelta = deltaLive(*bestIter);

      for (auto i = std::next(ready.cbegin(), 1); i != ready.cend(); ++i) {
        auto candidateDelta = deltaLive(*i);
        if (candidateDelta < bestDelta) {
          bestIter  = i;
          bestDelta = candidateDelta;
        }
      }
      auto address = *bestIter;
      sch.push_back(address);
      ready.erase(bestIter);
      for (auto allocAddress : g.getOp(address).getAllocs()) {
        --nOutstandingForAlloc[allocAddress];
        if (nOutstandingForAlloc[allocAddress] == 0) {
          allocLive[allocAddress] = false;
        } else {
          allocLive[allocAddress] = true;
        }
      }

      for (auto cAddress : g.getOp(address).getOuts()) {
        --outstanding[cAddress];
        if (outstanding[cAddress] == 0) {
          ready.push_back(cAddress);
        }
      }
    }
  }

  else {
    throw error("unrecognised KahnTieBreaker");
  }

  if (sch.size() != g.nOps()) {
    log().info("Failed to schedule all Ops, obtaining summary.");

    std::vector<std::string> dbs;
    dbs.reserve(g.nOps());
    for (uint64_t i = 0; i < g.nOps(); ++i) {
      dbs.push_back(g.getOp(i).getDebugString());
    }

    std::ostringstream oss;
    oss << "Only " << sch.size() << " of " << g.nOps()
        << " were scheduled, there is a cycle in the Graph."
        << " The non-singleton strongly connected components, "
        << "in topological order, are:"
        << scc::getSummary(
               g.getFwdEdges_u64(), dbs, scc::IncludeSingletons::No);

    throw error(oss.str());
  }

  return sch;
}

//
// Sets `schToOp` from the merged child graph of `merged`.
//
// For each op in the child schedule, looks up the ops it was merged from in
// the parent graph, using `childToParents` from `merged`, and adds them to
// the parent schedule.
std::vector<OpAddress> ScheduledGraph::getScheduleFromMergedChild(
    const Graph::OpMerged &merged,
    const std::vector<OpAddress> &childSchedule) {

  std::vector<OpAddress> sch;
  const auto &childGraph     = std::get<0>(merged);
  const auto &childToParents = std::get<1>(merged);

  for (ScheduleIndex i = 0; i < childGraph.nOps_i32(); ++i) {
    const auto childAddress = childSchedule.at(i);
    sch.insert(sch.end(),
               childToParents[childAddress].cbegin(),
               childToParents[childAddress].cend());
  }

  return sch;
}

std::vector<OpAddress> ScheduledGraph::kahn(const Graph &graph,
                                            const KahnTieBreaker kahnTie,
                                            const uint32_t seed) {
  auto opsWithFwdLinks = graph.getOpsWithFwdLinks();
  if (!opsWithFwdLinks.empty()) {
    auto merged      = graph.getLinkMerged();
    auto &childGraph = std::get<0>(merged);

    auto childSchedule = linklessKahn(childGraph, kahnTie, seed);
    return getScheduleFromMergedChild(merged, childSchedule);

  } else {
    return linklessKahn(graph, kahnTie, seed);
  }
}

bool ScheduledGraph::linklessIsSchedulable(const Graph &g) {

  std::vector<OpAddress> outstanding;
  outstanding.reserve(g.nOps());
  std::vector<OpAddress> ready;
  for (OpAddress i = 0; i < g.nOps(); ++i) {
    outstanding.push_back(g.getOp(i).nIns());
    if (outstanding[i] == 0) {
      ready.push_back(i);
    }
  }

  int nScheduled{0};

  while (!ready.empty()) {
    OpAddress address = ready.back();
    ++nScheduled;
    ready.pop_back();
    for (auto cAddress : g.getOp(address).getOuts()) {
      --outstanding[cAddress];
      if (outstanding[cAddress] == 0) {
        ready.push_back(cAddress);
      }
    }
  }

  return nScheduled == g.nOps_i32();
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

void ScheduledGraph::applyChange(const ScheduleChange &scheduleChange) {

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

void ScheduledGraph::setSchToLiveness() {
  schToLiveness.reserve(nOps());
  schToLiveness.clear();
  auto deltaLiveness = getDeltaLiveness();
  schToLiveness.push_back(deltaLiveness[0]);
  for (uint64_t i = 0; i < nOps(); ++i) {
    schToLiveness.push_back(schToLiveness.back() + deltaLiveness[i + 1]);
  }
}

bool ScheduledGraph::isSchedulable(const Graph &g) {

  const auto opsWithFwdLinks = g.getOpsWithFwdLinks();
  if (!opsWithFwdLinks.empty()) {
    const auto merged      = g.getLinkMerged();
    const auto &childGraph = std::get<0>(merged);
    return linklessIsSchedulable(childGraph);
  } else {
    return linklessIsSchedulable(g);
  }
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

  std::vector<std::string> sIndex{"Index", "====="};
  std::vector<std::string> sIns{"Ins", "==="};
  std::vector<std::string> sLinkTo{"LinkTo", "======"};
  std::vector<std::string> sOuts{"Outs", "===="};
  std::vector<std::string> sAllocs{"Allocs", "======="};
  std::vector<std::string> sLiveness{"Liveness", "========"};
  std::vector<std::string> sName{"Name", "===="};

  auto spcStr = [](uint64_t provision, const std::string &x) {
    uint64_t l = provision > x.size() ? provision - x.size() : 1;
    return std::string(l, ' ');
  };

  auto getProvision = [](const std::vector<std::string> &x) {
    return 1 + std::min<uint64_t>(
                   30, // never provision more space than this
                   std::accumulate(x.cbegin(),
                                   x.cend(),
                                   0UL,
                                   [](uint64_t a, const std::string &b) {
                                     return std::max<uint64_t>(a, b.size());
                                   }));
  };

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

  uint64_t provIndex    = getProvision(sIndex);
  uint64_t provLiveness = getProvision(sLiveness);
  uint64_t provIns      = getProvision(sIns);
  uint64_t provLinkTo   = getProvision(sLinkTo);
  uint64_t provOuts     = getProvision(sOuts);
  uint64_t provAllocs   = getProvision(sAllocs);
  uint64_t provName     = getProvision(sName);

  std::ostringstream oss;
  for (uint64_t i = 0; i < sIndex.size(); ++i) {
    oss << sIndex[i] << spcStr(provIndex, sIndex[i])          //
        << sName[i] << spcStr(provName, sName[i])             //
        << sIns[i] << spcStr(provIns, sIns[i])                //
        << sLinkTo[i] << spcStr(provLinkTo, sLinkTo[i])       //
        << sOuts[i] << spcStr(provOuts, sOuts[i])             //
        << sAllocs[i] << spcStr(provAllocs, sAllocs[i])       //
        << sLiveness[i] << spcStr(provLiveness, sLiveness[i]) //
        << '\n';
  }

  AllocWeight total = std::accumulate(
      schToLiveness.cbegin(), schToLiveness.cend(), AllocWeight(0));
  oss << "Total : " << total << '\n';

  return oss.str();
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

ScheduledGraph::ScheduledGraph(Graph &&g,
                               const Settings &settings,
                               const SolutionCache *readCache,
                               SolutionCache *writeCache) {

  if (readCache) {
    auto found = readCache->find(g, settings);
    if (found) {
      fromCache        = true;
      graph            = std::move(g);
      const auto &soln = *found;
      for (uint64_t i = 1; i < soln.size(); ++i) {
        graph.insertConstraint(soln[i - 1], soln[i]);
      }
      initialize(KahnTieBreaker::FIFO,
                 1011,
                 TransitiveClosureOptimizations::allOff());
    }
  }

  if (!fromCache) {
    graph   = std::move(g);
    auto g0 = graph;
    initialize(settings.kahnTieBreaker(), settings.seed(), settings.tcos());
    greedyRotate(settings.rotationAlgo(),
                 settings.debugMode(),
                 settings.seed(),
                 settings.rotationTermination());
    if (writeCache) {
      writeCache->writeSolution(std::move(g0), settings, schToOp);
    }
  }
}

ScheduledGraph::ScheduledGraph(Graph &&g,
                               const std::map<std::string, std::string> &m,
                               const SolutionCache *readCache,
                               SolutionCache *writeCache)
    : ScheduledGraph(std::move(g), Settings(m), readCache, writeCache) {}

ScheduledGraph::ScheduledGraph(Graph &&g,
                               const KahnTieBreaker ktb,
                               const TransitiveClosureOptimizations tco,
                               const RotationTermination rt,
                               const RotationAlgo algo,
                               const uint32_t seed,
                               const DebugMode debugMode,
                               const SolutionCache *readCache,
                               SolutionCache *writeCache)
    : ScheduledGraph(std::move(g),
                     Settings(ktb, tco, rt, algo, seed, debugMode),
                     readCache,
                     writeCache) {}

void ScheduledGraph::greedyRotate(RotationAlgo algo,
                                  DebugMode debugMode,
                                  uint32_t seed,
                                  RotationTermination rt) {

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

  int nChangesAtCurrentShift{0};

  // at a given shift, there may be multiple rounds
  int nChangesInCurrentRound{0};
  AllocWeight deltaWeightCurrentRound{0};

  // there have been no-recorded improvements since last at nToShift = 1
  bool noChangeSinceStart{true};

  // one "shift" phase can consist of multple "rounds"
  double timeSpentInCurrentRound{0.0};
  double timeSpentInTotal{0.0};

  int64_t nChangesInTotal{0};

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

        applyChange(scheduleChange);

        if (debugMode == DebugMode::On) {
          assertCorrectness();
        }
        ++nChangesInCurrentRound;
        deltaWeightCurrentRound += shiftAndCost.getCost();
        totalDeltaSumLiveness += shiftAndCost.getCost();
      }
    }

    nChangesAtCurrentShift += nChangesInCurrentRound;
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
    oss << "nChangesInCurrentRound = " << nChangesInCurrentRound << " ";

    if (noChangeSinceStart) {
      oss << "noChangeSinceStart, so " << nToShift << " --> " << nToShift + 1;
      ++nToShift;
      resetSusceptibleTrue();
    } else if (nChangesInCurrentRound == 0) {
      oss << "no changes, so " << nToShift << " -->  1, cleanSlate";
      nToShift           = 1;
      noChangeSinceStart = true;
      resetSusceptibleTrue();
    } else {
      oss << "staying at " << nToShift;
      nToShift = oldNToShift;
    }

    log().info(oss.str());

    if (oldNToShift != nToShift) {
      updateCanCan(oldNToShift, nToShift);
      nChangesAtCurrentShift  = 0;
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
