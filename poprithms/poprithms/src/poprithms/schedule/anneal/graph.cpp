#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

namespace {

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
} // namespace

OpAddress Graph::insertOp(const std::string &dbs) {
  OpAddress op = nOps();
  allOps.push_back({op, dbs});
  return op;
}

std::vector<OpAddress>
Graph::insertOps(const std::vector<std::string> &dbStrings) {
  std::vector<OpAddress> opAdds;
  opAdds.reserve(dbStrings.size());
  for (const auto &dbs : dbStrings) {
    opAdds.push_back(insertOp(dbs));
  }
  return opAdds;
}

std::vector<std::array<OpAddress, 2>> Graph::getTightPairs() const {

  std::vector<std::array<OpAddress, 2>> tightPairs;
  for (const auto &op : getOps()) {
    if (op.nOuts() == 1UL && getOp(op.getOut(0)).nIns() == 1UL) {
      tightPairs.push_back(
          {op.getAddress(), getOp(op.getOut(0)).getAddress()});
    }
  }
  return tightPairs;
}

void Graph::insertOpAlloc(OpAddress oa, AllocAddress aa) {
  allAllocs[aa].insertOp(oa);
  allOps[oa].insertAlloc(aa);
}

void Graph::insertOpAlloc(const std::vector<OpAddress> &oas,
                          AllocAddress aa) {
  for (auto oa : oas) {
    insertOpAlloc(oa, aa);
  }
}

void Graph::insertBinConstraints(
    const std::vector<std::vector<OpAddress>> &bins,
    const std::string &prefix) {
  for (uint64_t i = 1; i < bins.size(); ++i) {
    // a "bottleneck" Op, which partitions Ops into different bins.
    auto op = insertOp(prefix + std::to_string(i));
    for (auto b : bins[i - 1]) {
      insertConstraint(b, op);
    }
    for (auto a : bins[i]) {
      insertConstraint(op, a);
    }
  }
}

void Graph::insertAttractions(
    const std::vector<std::array<OpAddress, 2>> &knots,
    AllocWeight w) {
  for (const auto &knot : knots) {
    auto allocAddress = insertAlloc(w);
    insertOpAlloc(std::get<0>(knot), allocAddress);
    insertOpAlloc(std::get<1>(knot), allocAddress);
  }
}

void Graph::insertConstraint(OpAddress before, OpAddress after) {
  allOps[before].insertOut(after);
  allOps[after].insertIn(before);
}

void Graph::insertLink(OpAddress before, OpAddress after) {
  insertConstraint(before, after);

  const auto &op0 = getOp(before);
  const auto &op1 = getOp(after);

  if (op0.hasForwardLink() && op0.getForwardLink() != after) {
    std::ostringstream oss;
    oss << "Ops can have at most one link forward. "
        << "Op " << op0 << " already has " << getOp(op0.getForwardLink())
        << " as a forward link, and so " << op1
        << " cannot be added as a forward link.";
    throw error(oss.str());
  }

  if (op1.hasBackwardLink() && op1.getBackwardLink() != before) {
    std::ostringstream oss;
    oss << "Ops can have at most one link backward. "
        << "Op " << op1 << " already has " << getOp(op1.getBackwardLink())
        << " as a backward link, and so " << op0
        << " cannot be added as a backward link.";
    throw error(oss.str());
  }

  if (!op0.hasForwardLink()) {
    opsWithFwdLinks.push_back(before);
  }
  allOps[before].insertForwardLink(after);
  allOps[after].insertBackwardLink(before);
}

void Graph::insertConstraints(
    const std::vector<std::array<OpAddress, 2>> &cs) {
  for (const auto &c : cs) {
    insertConstraint(std::get<0>(c), std::get<1>(c));
  }
}

void Graph::append(std::ostream &ost) const {
  for (auto op : getOps()) {
    ost << '\n' << op.getDebugString() << "   <-  [";
    for (auto inAdd : op.getIns()) {
      ost << ' ' << getOp(inAdd).getDebugString() << ' ';
    }
    ost << ']';
  }
}

std::vector<AllocWeight> Graph::getFwdRippleCosts(ScheduleIndex start0,
                                                  int nToShift,
                                                  int firstExtCon) const {

  const int sign             = +1;
  const auto nCostsToCompute = firstExtCon - nToShift - start0;
  const auto dirOffset       = nToShift - 1;
  return getRippleCosts(start0, nToShift, sign, nCostsToCompute, dirOffset);
}

AllocWeight Graph::getShiftCost(ScheduleIndex start0,
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

ScheduleIndex Graph::getLastProducer(ScheduleIndex start,
                                     int nToShift) const {

  //                       ......xxxxxxxxxx......
  // producers of all xs:   |  | |   |   ||
  //                            |
  //                       last producer

  // if there is no producer, this is what is returned:
  ScheduleIndex lower = -1;

  for (ScheduleIndex i = start; i < start + nToShift; ++i) {
    OpAddress opAddress = scheduleToOp(i);
    auto firstFromStart =
        custom_lower_bound(opToInSchedule(opAddress).cbegin(),
                           opToInSchedule(opAddress).cend(),
                           start);
    if (firstFromStart != opToInSchedule(opAddress).cbegin()) {
      lower = std::max(lower, *(std::prev(firstFromStart)));
    }
  }

  return lower;
}

ScheduleIndex Graph::getFirstConsumer(ScheduleIndex start,
                                      int nToShift) const {

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

ShiftAndCost Graph::getBestShiftRippleAlgo(const ScheduleIndex start,
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
      if (cost < bestCost) {
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
      if (fwdCosts[i] < bestCost) {
        bestCost  = fwdCosts[i];
        bestShift = shift;
      }
    }
  }

  ShiftAndCost best{bestShift, bestCost};
  return best;
}

std::vector<AllocAddress> Graph::getAllocAddresses(ScheduleIndex start,
                                                   ScheduleIndex end) const {
  std::vector<AllocAddress> addresses;
  auto nAddressEstimate = static_cast<uint64_t>(2 * (end - start));
  addresses.reserve(nAddressEstimate);
  for (ScheduleIndex scheduleIndex = start; scheduleIndex < end;
       ++scheduleIndex) {
    for (AllocAddress allocAddress : scheduleToAllocs(scheduleIndex)) {
      addresses.push_back(allocAddress);
    }
  }
  std::sort(addresses.begin(), addresses.end());

  // profiling reveals this is faster than the map -> vector approach
  auto last = std::unique(addresses.begin(), addresses.end());
  addresses.erase(last, addresses.cend());
  return addresses;
}

std::vector<OpAddress> Graph::getInputOps() const {
  std::vector<OpAddress> inputs;
  for (const auto &op : allOps) {
    if (op.nIns() == 0) {
      inputs.push_back(op.getAddress());
    }
  }
  return inputs;
}

std::string Graph::getLivenessString() const {

  std::vector<std::string> sIndex{"Index", "====="};
  std::vector<std::string> sIns{"Ins", "==="};
  std::vector<std::string> sOuts{"Outs", "===="};
  std::vector<std::string> sAllocsIn{"+Allocs", "======="};
  std::vector<std::string> sAllocsOut{"-Allocs", "======="};
  std::vector<std::string> sAllocs{"Allocs", "======="};
  std::vector<std::string> sLiveness{"Liveness", "========"};
  std::vector<std::string> sName{"Name", "===="};

  auto spaceString = [](uint64_t provision, const std::string &x) {
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

    auto address = schToOp[i];

    std::ostringstream ossIns;
    poprithms::util::append(ossIns, opToInSch[address]);
    std::ostringstream ossName;
    ossName << getOp(address).getDebugString();
    std::ostringstream ossOuts;
    poprithms::util::append(ossOuts, opToOutSch[address]);
    std::ostringstream ossAllocsIn;
    poprithms::util::append(ossAllocsIn, schToAllocFirsts[i]);
    std::ostringstream ossAllocsOut;
    poprithms::util::append(ossAllocsOut, schToAllocFinals[i]);
    std::ostringstream ossAllocs;
    poprithms::util::append(ossAllocs, schToAllocs[i]);

    sIndex.push_back(std::to_string(i));
    sLiveness.push_back(toString(schToLiveness[i]));
    sIns.push_back(ossIns.str());
    sOuts.push_back(ossOuts.str());
    sAllocsIn.push_back(ossAllocsIn.str());
    sAllocsOut.push_back(ossAllocsOut.str());
    sAllocs.push_back(ossAllocs.str());
    sName.push_back(ossName.str());
  }

  uint64_t provIndex     = getProvision(sIndex);
  uint64_t provLiveness  = getProvision(sLiveness);
  uint64_t provIns       = getProvision(sIns);
  uint64_t provOuts      = getProvision(sOuts);
  uint64_t provAllocsIn  = getProvision(sAllocsIn);
  uint64_t provAllocsOut = getProvision(sAllocsOut);
  uint64_t provAllocs    = getProvision(sAllocs);
  uint64_t provName      = getProvision(sName);

  std::ostringstream oss;
  for (uint64_t i = 0; i < sIndex.size(); ++i) {
    oss << sIndex[i] << spaceString(provIndex, sIndex[i])             //
        << sName[i] << spaceString(provName, sName[i])                //
        << sIns[i] << spaceString(provIns, sIns[i])                   //
        << sOuts[i] << spaceString(provOuts, sOuts[i])                //
        << sAllocsIn[i] << spaceString(provAllocsIn, sAllocsIn[i])    //
        << sAllocsOut[i] << spaceString(provAllocsOut, sAllocsOut[i]) //
        << sAllocs[i] << spaceString(provAllocs, sAllocs[i])          //
        << sLiveness[i] << spaceString(provLiveness, sLiveness[i])    //
        << '\n';
  }

  AllocWeight total = std::accumulate(
      schToLiveness.cbegin(), schToLiveness.cend(), AllocWeight(0));
  oss << "Total : " << total << '\n';

  return oss.str();
}

AllocAddress Graph::insertAlloc(AllocWeight w) {
  AllocAddress a = allAllocs.size();
  allAllocs.push_back({a, w});
  return a;
}

template <typename T>
void rotate(T &t, ScheduleIndex x0, ScheduleIndex o0, ScheduleIndex o1) {
  auto t0 = std::next(t.begin(), x0);
  auto t1 = std::next(t.begin(), o0);
  auto t2 = std::next(t.begin(), o1);
  std::rotate(t0, t1, t2);
}

void Graph::applyChange(const ScheduleChange &scheduleChange) {

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

  std::vector<OpAddress> consumersTouched;
  auto estimateOfNEdges = static_cast<uint64_t>(2 * (o1 - x0));
  consumersTouched.reserve(estimateOfNEdges);
  for (ScheduleIndex i = x0; i < o1; ++i) {
    for (auto outAddress : getOp(scheduleToOp(i)).getOuts()) {
      consumersTouched.push_back(outAddress);
    }
  }
  std::sort(consumersTouched.begin(), consumersTouched.end());
  auto lastConsumer =
      std::unique(consumersTouched.begin(), consumersTouched.end());
  consumersTouched.erase(lastConsumer, consumersTouched.end());

  std::vector<OpAddress> producersTouched;
  producersTouched.reserve(estimateOfNEdges);
  for (ScheduleIndex i = x0; i < o1; ++i) {
    for (auto inAddress : getOp(scheduleToOp(i)).getIns()) {
      producersTouched.push_back(inAddress);
    }
  }
  std::sort(producersTouched.begin(), producersTouched.end());
  auto lastProducer =
      std::unique(producersTouched.begin(), producersTouched.end());
  producersTouched.erase(lastProducer, producersTouched.end());

  // 4 opToInSch
  for (OpAddress consumerAddress : consumersTouched) {
    setOpToInSch(consumerAddress);
  }

  // 5 opToOutSch
  for (OpAddress producerAddress : producersTouched) {
    setOpToOutSch(producerAddress);
  }

  // 6 schToAllocFirsts
  rotate(schToAllocFirsts, x0, o0, o1);

  // 7 schToAllocFinals
  rotate(schToAllocFinals, x0, o0, o1);

  // 9 nCanFwd and nCanBwd
  updateNCanFwds(nToShift, x0, o1, producersTouched);
  updateNCanBwds(nToShift, x0, o1, consumersTouched);
}

// TODO(T14827) : there's a faster way to do this when n2s = 1, will be useful
// when multi-threading as fast updating will become important
void Graph::updateNCanFwds(int n2s,
                           int x0,
                           int o1,
                           const std::vector<OpAddress> &producersTouched) {

  // for all x in ScheduleIndices, update how far forward the range [x, x+
  // n2s) can shift forwards

  // determine which schedule indices may have changed
  ScheduleIndex startCanFwdRange = x0;
  ScheduleIndex endCanFwdRange   = o1;
  for (OpAddress p : producersTouched) {
    startCanFwdRange = std::min(startCanFwdRange, opToSch[p]);
  }
  startCanFwdRange -= n2s + 1;
  startCanFwdRange = std::max(0, startCanFwdRange);
  endCanFwdRange   = std::min(endCanFwdRange, nOps_i32() - n2s + 1);

  // reset the schedule indices which may have changed
  for (ScheduleIndex i = startCanFwdRange; i < endCanFwdRange; ++i) {
    auto i_u64     = static_cast<uint64_t>(i);
    nCanFwd[i_u64] = getFirstConsumer(i, n2s) - i - n2s;
  }
}

// TODO(T14827) : there's a faster way to do this when n2s = 1.
void Graph::updateNCanBwds(int n2s,
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

void Graph::setSchToLiveness() {
  schToLiveness.reserve(nOps());
  schToLiveness.clear();
  auto deltaLiveness = getDeltaLiveness();
  schToLiveness.push_back(deltaLiveness[0]);
  for (uint64_t i = 0; i < nOps(); ++i) {
    schToLiveness.push_back(schToLiveness.back() + deltaLiveness[i + 1]);
  }
}

bool Graph::isSchedulable() const {

  if (hasAtLeastOneLink()) {
    auto merged      = getLinkMerged();
    auto &childGraph = std::get<0>(merged);
    return childGraph.isSchedulable();
  }

  if (!isFinalized) {
    throw error(
        "Graph not finalized, should call finalize() before isSchedulable()");
  }

  std::vector<OpAddress> outstanding;
  outstanding.reserve(nOps());
  std::vector<OpAddress> ready;
  for (OpAddress i = 0; i < allOps.size(); ++i) {
    outstanding.push_back(getOp(i).nIns());
    if (outstanding[i] == 0) {
      ready.push_back(i);
    }
  }

  int nScheduled{0};

  while (!ready.empty()) {
    OpAddress address = ready.back();
    ++nScheduled;
    ready.pop_back();
    for (auto cAddress : allOps[address].getOuts()) {
      --outstanding[cAddress];
      if (outstanding[cAddress] == 0) {
        ready.push_back(cAddress);
      }
    }
  }

  return nScheduled == nOps_i32();
}

Graph::LinkMerged Graph::getLinkMerged() const {

  Graph childGraph;

  const auto chains = getLinkChains();

  // The Allocs are the same in the child Graph as the parent Graph
  for (const auto &parentAlloc : getAllocs()) {
    childGraph.insertAlloc(parentAlloc.getWeight());
  }

  // Map an Op in the parent Graph to its unique Op in the child Graph
  std::vector<OpAddress> parentToChild(nOps(), 0);

  // We assign lowest addresses to child Ops which are generated from parent
  // chains, then the remaining addresses are assigned to the unchained Ops
  uint64_t childOpAddress{0};
  while (childOpAddress < chains.size()) {
    for (const auto opAddress : chains[childOpAddress]) {
      parentToChild[opAddress] = childOpAddress;
    }
    ++childOpAddress;
  }

  // Map an Op in the child Graph to its parent(s) in the parent Graph
  ParentGraphOps childToParents = std::move(chains);

  for (uint64_t parentAddress = 0; parentAddress < nOps(); ++parentAddress) {
    if (!getOp(parentAddress).hasLink()) {
      parentToChild[parentAddress] = childOpAddress;
      ++childOpAddress;
      childToParents.push_back({parentAddress});
    }
  }

  const auto nChildOps = childOpAddress;

  for (uint64_t childAddress = 0; childAddress < nChildOps; ++childAddress) {
    // The child Op's name is a concatenation of the names of the parent Ops
    const auto &parentAddresses = childToParents[childAddress];
    std::ostringstream ossChildName;
    ossChildName << '(';
    for (uint64_t i = 0; i < parentAddresses.size(); ++i) {
      if (i != 0) {
        ossChildName << ' ';
      }
      ossChildName << getOp(parentAddresses[i]).getDebugString();
    }
    ossChildName << ')';
    auto name = ossChildName.str();
    childGraph.insertOp(name);
  }

  for (uint64_t childAddress = 0; childAddress < nChildOps; ++childAddress) {

    // child Op inherits constraints and Allocs from parent(s)
    for (auto parentAddress : childToParents[childAddress]) {
      const auto &parent = getOp(parentAddress);
      for (auto allocAddress : parent.getAllocs()) {
        childGraph.insertOpAlloc(childAddress, allocAddress);
      }
      for (auto outParentAddress : parent.getOuts()) {
        auto outChildAddress = parentToChild[outParentAddress];
        if (outChildAddress != childAddress) {
          childGraph.insertConstraint(childAddress, outChildAddress);
        }
      }
    }
  }

  childGraph.finalize();

  return {childGraph, childToParents};
}

std::vector<std::vector<OpAddress>> Graph::getLinkChains() const {

  std::vector<std::vector<OpAddress>> chains;

  for (auto address : opsWithFwdLinks) {
    // start of a chain
    if (!getOp(address).hasBackwardLink()) {
      chains.push_back({});
      auto current = address;
      while (getOp(current).hasForwardLink()) {
        chains.back().push_back(current);
        current = getOp(current).getForwardLink();
      }
      chains.back().push_back(current);
    }
  }

  return chains;
}

void Graph::khan(KhanTieBreaker khanTie, uint32_t khanSeed) {

  schToOp.reserve(nOps());
  schToOp.clear();

  if (hasAtLeastOneLink()) {
    auto merged                = getLinkMerged();
    auto &childGraph           = std::get<0>(merged);
    const auto &childToParents = std::get<1>(merged);
    childGraph.khan(khanTie, khanSeed);
    for (ScheduleIndex i = 0; i < childGraph.nOps(); ++i) {
      const auto childAddress = childGraph.scheduleToOp(i);
      schToOp.insert(schToOp.end(),
                     childToParents[childAddress].cbegin(),
                     childToParents[childAddress].cend());
    }
    return;
  }

  std::vector<OpAddress> outstanding;
  outstanding.reserve(nOps());
  std::vector<OpAddress> ready;
  for (OpAddress i = 0; i < allOps.size(); ++i) {
    outstanding.push_back(getOp(i).nIns());
    if (outstanding[i] == 0) {
      ready.push_back(i);
    }
  }

  std::mt19937 g(khanSeed);

  if (khanTie == KhanTieBreaker::RANDOM) {
    while (!ready.empty()) {
      std::shuffle(ready.begin(), ready.end(), g);
      OpAddress address = ready.back();
      schToOp.push_back(address);
      ready.pop_back();
      for (auto cAddress : allOps[address].getOuts()) {
        --outstanding[cAddress];
        if (outstanding[cAddress] == 0) {
          ready.push_back(cAddress);
        }
      }
    }
  }

  else if (khanTie == KhanTieBreaker::FIFO) {
    while (!ready.empty()) {
      OpAddress address = ready.back();
      schToOp.push_back(address);
      ready.pop_back();
      for (auto cAddress : allOps[address].getOuts()) {
        --outstanding[cAddress];
        if (outstanding[cAddress] == 0) {
          ready.push_back(cAddress);
        }
      }
    }
  }

  // GREEDY
  else if (khanTie == KhanTieBreaker::GREEDY) {

    std::vector<int> nOutstandingForAlloc(nAllocs());
    std::vector<bool> allocLive(nAllocs(), false);
    for (const auto &alloc : getAllocs()) {
      nOutstandingForAlloc[alloc.getAddress()] = alloc.nOps_i32();
    }

    auto deltaLive = [&nOutstandingForAlloc, &allocLive, this](OpAddress a) {
      AllocWeight delta{0};
      for (auto allocAddress : getOp(a).getAllocs()) {
        const auto allocWeight = getAlloc(allocAddress).getWeight();
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
      std::shuffle(ready.begin(), ready.end(), g);

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
      schToOp.push_back(address);
      ready.erase(bestIter);
      for (auto allocAddress : getOp(address).getAllocs()) {
        --nOutstandingForAlloc[allocAddress];
        if (nOutstandingForAlloc[allocAddress] == 0) {
          allocLive[allocAddress] = false;
        } else {
          allocLive[allocAddress] = true;
        }
      }

      for (auto cAddress : allOps[address].getOuts()) {
        --outstanding[cAddress];
        if (outstanding[cAddress] == 0) {
          ready.push_back(cAddress);
        }
      }
    }
  }

  else {
    throw error("unrecognised KhanTieBreaker");
  }

  if (schToOp.size() != allOps.size()) {
    std::ostringstream oss;
    oss << "Failed to schedule Ops in Graph::initializeSchedule, "
        << " only managed to schedule " << schToOp.size() << " of "
        << allOps.size() << ". Failed to schedule:\n";

    for (const auto &op : allOps) {
      if (std::find(schToOp.cbegin(), schToOp.cend(), op.getAddress()) ==
          schToOp.cend()) {
        oss << "     " << op.getAddress() << " <- { ";
        for (auto x : op.getIns()) {
          oss << x << ' ';
        }
        oss << "}   [  " << op << "  ] \n";
      }
    }
    throw error(oss.str());
  }
}

void Graph::finalize() {
  for (auto &op : allOps) {
    op.sortAndMakeUnique();
  }
  for (auto &alloc : allAllocs) {
    alloc.sortAndMakeUnique();
  }
  isFinalized = true;
}

void Graph::initialize(KhanTieBreaker khanTie, uint32_t khanSeed) {

  if (!isFinalized) {
    finalize();
  }

  //
  // schToOp. Vanilla run of Khan's O(E) algorithm, random tie-breaks
  khan(khanTie, khanSeed);

  //
  // opToSch
  opToSch.reserve(nOps());
  opToSch.clear();

  for (ScheduleIndex i = 0; i < nOps(); ++i) {
    opToSch[scheduleToOp(i)] = i;
  }

  //
  // allocToSch
  allocToSch.resize(nAllocs());
  allocToSch.clear();

  for (AllocAddress allocAddress = 0; allocAddress < nAllocs();
       ++allocAddress) {
    setAllocToSch(allocAddress);
  }

  //
  // schToAllocs
  schToAllocs.reserve(nOps());
  schToAllocs.clear();

  for (ScheduleIndex schedIndex = 0; schedIndex < nOps(); ++schedIndex) {
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
  // schToAllocFirsts, schToAllocFinals
  schToAllocFirsts.clear();
  schToAllocFirsts.resize(nOps());
  schToAllocFinals.clear();
  schToAllocFinals.resize(nOps());
  for (AllocAddress allocAddress = 0; allocAddress < nAllocs();
       ++allocAddress) {
    if (getAlloc(allocAddress).nOps() > 0) {
      auto firstSched     = allocToFirstSchedule(allocAddress);
      auto firstSched_u64 = static_cast<uint64_t>(firstSched);
      auto finalSched     = allocToFinalSchedule(allocAddress);
      auto finalSched_u64 = static_cast<uint64_t>(finalSched);
      schToAllocFirsts[firstSched_u64].push_back(allocAddress);
      schToAllocFinals[finalSched_u64].push_back(allocAddress);
    }
  }

  for (auto &x : schToAllocFirsts) {
    std::sort(x.begin(), x.end());
  }
  for (auto &x : schToAllocFinals) {
    std::sort(x.begin(), x.end());
  }

  //
  // schToLiveness
  setSchToLiveness();

  //
  // nFwd, nBwd
  setCanCan(1);

  rippleScratch.resize(
      nAllocs(),
      {-1, AllocWeight::negativeOne(), AllocWeight::negativeOne(), false});

  isInitialized = true;
}

void Graph::setCanCan(int nToShift) {

  nCanFwd.clear();
  auto numNCan = static_cast<uint64_t>(nOps_i32() - nToShift + 1);
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

void Graph::updateCanCan(int oldNToShift, int n2s) {

  // bootstrapping off oldNToShift is significantly faster
  if (n2s - oldNToShift == 1) {

    // with an increase of 1 of nToShift, the number of possible starts
    // decreases by 1
    nCanFwd.pop_back();
    nCanBwd.pop_back();

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

std::vector<AllocWeight> Graph::getDeltaLiveness() const {
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

void Graph::assertCorrectness() const {

  if (!isInitialized) {
    throw error(
        "assertCorrectness() should only be called after initialize(.,.)");
  }

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
      if (opToSch[j] >= i) {
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
}

ShiftAndCost Graph::getBestShiftSimpleAlgo(const ScheduleIndex start0,
                                           const int nToShift) const {

  // sum over Allocs of allocWeight * ( liveness duration )
  auto getTotal = [this](const std::vector<ScheduleIndex> &o2s) {
    AllocWeight tot{0};
    for (const auto &alloc : getAllocs()) {
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

    if (newTotal < bestTotal) {
      bestTotal = newTotal;
      bestShift = start1 - start0;
    }
  }

  return {bestShift, bestTotal - currentTotal};
}

namespace {

std::string getMinSumLivenessAlgoString(MinSumLivenessAlgo algo) {
  switch (algo) {
  case (MinSumLivenessAlgo::SIMPLE): {
    return "Simple";
  }
  case (MinSumLivenessAlgo::RIPPLE): {
    return "Ripple";
  }
  default: {
    throw error("Unrecognised MinSumLivenessAlgo");
  }
  }
}
} // namespace

void Graph::confirmShiftAndCost(ScheduleIndex start0,
                                int nToShift,
                                const ShiftAndCost &shiftAndCost,
                                MinSumLivenessAlgo algo) const {

  auto debugShiftAndCost = getBestShiftSimpleAlgo(start0, nToShift);
  if (debugShiftAndCost != shiftAndCost) {
    const std::string runAlgoStr = getMinSumLivenessAlgoString(algo);
    const std::string basicAlgoStr =
        getMinSumLivenessAlgoString(MinSumLivenessAlgo::SIMPLE);
    std::ostringstream oss;
    oss << "\nDifferent results between " << runAlgoStr << " and "
        << basicAlgoStr << " in debug for minSumLivenessAnneal. "
        << "The current state of schedule is \n"
        << getLivenessString() << "With " << basicAlgoStr
        << ", the suggestion is\n  " << debugShiftAndCost << "\nand with"
        << runAlgoStr << ", the suggestion is\n  " << shiftAndCost
        << ".\nThis for start0 = " << start0
        << " and nToShift = " << nToShift;
    throw error(oss.str());
  }
}

void Graph::minSumLivenessAnneal(
    const std::map<std::string, std::string> &m) {
  bool debug               = defaultDebug();
  uint32_t seed            = defaultSeed();
  Fraction pStayPut        = defaultPStayPut();
  Fraction pHigherFallRate = defaultPHigherFallRate();
  Fraction pClimb          = defaultPClimb();
  bool logging             = defaultLogging();
  double timeLimitSeconds  = defaultTimeLimitSeconds();
  int64_t swapLimitCount   = defaultSwapLimitCount();

  for (auto &[k, v] : m) {
    if (k == "debug") {
      debug = static_cast<bool>(std::stoi(v));
    } else if (k == "seed") {
      seed = static_cast<uint32_t>(std::stoll(v));
    } else if (k == "pHigherFallRate") {
      pHigherFallRate = static_cast<Fraction>(std::stof(v));
    } else if (k == "pStayPut") {
      pStayPut = static_cast<Fraction>(std::stof(v));
    } else if (k == "pClimb") {
      pClimb = static_cast<Fraction>(std::stof(v));
    } else if (k == "logging") {
      logging = static_cast<bool>(std::stoi(v));
    } else if (k == "timeLimitSeconds") {
      timeLimitSeconds = static_cast<double>(std::stof(v));
    } else if (k == "swapLimitCount") {
      swapLimitCount = static_cast<int64_t>(std::stoll(v));
    } else {
      throw error("invalid option in minSumLivenessAnneal, " + k);
    }
  }
  minSumLivenessAnneal(MinSumLivenessAlgo::RIPPLE,
                       debug,
                       seed,
                       pStayPut,
                       pHigherFallRate,
                       pClimb,
                       logging,
                       timeLimitSeconds,
                       swapLimitCount);
}

void Graph::minSumLivenessAnneal(MinSumLivenessAlgo algo,
                                 bool debug,
                                 uint32_t seed,
                                 Fraction pStayPut,
                                 Fraction pHigherFallRate,
                                 Fraction pClimb,
                                 bool logging,
                                 double timeLimitSeconds,
                                 int64_t swapLimitCount) {

  if (logging) {
    std::cout << "debug=" << debug << " seed=" << seed
              << " pStayPuy=" << pStayPut
              << " pHigherFallRate=" << pHigherFallRate
              << " pClimb=" << pClimb
              << " timeLimitSeconds=" << timeLimitSeconds
              << " swapLimitCount=" << swapLimitCount << std::endl;
  }

  std::vector<FallRate> fallRates;
  std::mt19937 g(seed);

  if (pStayPut < 0.0) {
    throw error("pStayPut must be non-negative");
  }

  if (pHigherFallRate < 0.0) {
    throw error("pHigherClimbRate must be non-negative");
  }

  if (pClimb < 0.0) {
    throw error("pClimb must be non-negative");
  }

  auto pSum = pStayPut + pHigherFallRate + pClimb;
  if (pSum <= 0.0) {
    throw error(
        "pStayPut + pHigherFallRate + pClimb must be strictly positive");
  }

  std::uniform_real_distribution<> realDis(0.0, pSum);

  // look for moves of this shift length
  int nToShift{1};
  bool continueAnnealing =
      (timeLimitSeconds <= 0 || swapLimitCount <= 0) ? false : true;

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

  std::vector<ScheduleIndex> indices;

  auto updateIndices = [&indices, &nToShift, this]() {
    int nIndices          = nOps_i32() + 1 - nToShift;
    uint64_t nIndices_u64 = static_cast<uint64_t>(nIndices);
    indices               = std::vector<ScheduleIndex>(nIndices_u64);
    std::iota(indices.begin(), indices.end(), 0);
  };

  updateIndices();

  auto startCurrentShift = std::chrono::high_resolution_clock::now();

  // used in a check for the correctness of all computed improvements
  const AllocWeight initSumLiveness = getSumLiveness();
  const AllocWeight initMaxLiveness = getMaxLiveness();
  AllocWeight totalDeltaSumLiveness{0};

  while (continueAnnealing) {

    auto startCurrentRound = std::chrono::high_resolution_clock::now();

    // in each round. the order in which the indices are considered is
    // different. The idea is that this prevents bad cases analogous to bad
    // hash functions, although these haven't been obsereved at time of
    // commenting
    std::shuffle(indices.begin(), indices.end(), g);
    nChangesInCurrentRound  = 0;
    deltaWeightCurrentRound = AllocWeight::zero();
    for (auto start0 : indices) {
      ShiftAndCost shiftAndCost{-1, AllocWeight::negativeOne()};
      if (algo == MinSumLivenessAlgo::RIPPLE) {
        shiftAndCost = getBestShiftRippleAlgo(start0, nToShift);
      } else {
        shiftAndCost = getBestShiftSimpleAlgo(start0, nToShift);
      }
      if (debug) {
        confirmShiftAndCost(start0, nToShift, shiftAndCost, algo);
      }
      if (shiftAndCost.getCost() < AllocWeight(0)) {
        auto start1 = start0 + shiftAndCost.getShift();
        ScheduleChange scheduleChange{start0, start1, nToShift};
        applyChange(scheduleChange);
        if (debug) {
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
    if (timeSpentInTotal > timeLimitSeconds) {
      continueAnnealing = false;
    }
    if (nChangesInTotal >= swapLimitCount) {
      continueAnnealing = false;
    }

    auto fallRatesMinSize = static_cast<uint64_t>(nToShift + 1);
    if (fallRates.size() < fallRatesMinSize) {
      fallRates.resize(fallRatesMinSize, AllocWeight::zero());
    }

    uint64_t nToShift_u64 = static_cast<uint64_t>(nToShift);
    fallRates[nToShift_u64] =
        deltaWeightCurrentRound / timeSpentInCurrentRound;

    auto oldNToShift     = nToShift;
    auto oldNToShift_u64 = nToShift_u64;

    std::ostringstream oss;

    if (noChangeSinceStart) {
      oss << "noChangeSinceStart, so " << nToShift << " --> " << nToShift + 1;
      ++nToShift;

    } else if (nChangesInCurrentRound == 0) {
      oss << "no changes, so " << nToShift << " -->  1, cleaSlate";
      nToShift           = 1;
      noChangeSinceStart = true;
    } else {
      auto p = realDis(g);
      if (p < pStayPut) {
        oss << "staying at " << nToShift;
        nToShift = oldNToShift;
      } else if (p < pStayPut + pClimb) {
        oss << "climbing " << nToShift << " --> " << nToShift + 1;
        nToShift = oldNToShift + 1;
      } else {
        auto bestFallRate = std::accumulate(
            fallRates.cbegin(),
            std::next(fallRates.cbegin(), oldNToShift),
            FallRate(0.0),
            [](FallRate a, FallRate b) { return std::min(a, b); });
        oss << "fal rate at " << oldNToShift_u64 << " is "
            << fallRates[oldNToShift_u64] << " best fall rate in [1, "
            << oldNToShift_u64 << ") is " << bestFallRate << ": ";
        if (fallRates[oldNToShift_u64] < bestFallRate) {
          oss << "staying at " << oldNToShift_u64;
          nToShift = oldNToShift;

        } else {
          oss << " reset to 1";
          nToShift           = 1;
          noChangeSinceStart = true;
        }
      }
    }

    nToShift_u64 = static_cast<uint64_t>(nToShift);

    if (logging) {
      std::cout << oss.str() << std::endl;
    }

    if (oldNToShift != nToShift) {

      updateIndices();

      nChangesAtCurrentShift = 0;
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
        continueAnnealing = false;
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
      // if (finalSumLiveness - initSumLiveness != totalDeltaSumLiveness) {
      std::ostringstream oss2;
      oss2 << "An error might have occurred in minSumLivenessAnneal. "
           << "The running accumulation of calculated improvements is "
           << totalDeltaSumLiveness << '.' << ' '
           << "The difference between initial and final liveness sums is "
           << finalSumLiveness - initSumLiveness << '.';
      throw error(oss2.str());
    }
  }

  if (logging) {
    std::cout << "init sum liveness =  " << initSumLiveness << '\n'
              << "final sum liveness = " << finalSumLiveness << '.'
              << std::endl;

    std::cout << "init max liveness =  " << initMaxLiveness << '\n'
              << "final max liveness = " << finalMaxLiveness << '.'
              << std::endl;
  }
}

void Graph::setOpToInSch(OpAddress opAddress) {
  opToInSch[opAddress].reserve(getOp(opAddress).nIns());
  opToInSch[opAddress].clear();
  for (OpAddress inAddress : getOp(opAddress).getIns()) {
    opToInSch[opAddress].push_back(opToSch[inAddress]);
  }
  std::sort(opToInSch[opAddress].begin(), opToInSch[opAddress].end());
}

void Graph::setOpToOutSch(OpAddress opAddress) {
  opToOutSch[opAddress].reserve(getOp(opAddress).nOuts());
  opToOutSch[opAddress].clear();
  for (OpAddress outAddress : getOp(opAddress).getOuts()) {
    opToOutSch[opAddress].push_back(opToSch[outAddress]);
  }
  std::sort(opToOutSch[opAddress].begin(), opToOutSch[opAddress].end());
}

void Graph::setAllocToSch(AllocAddress allocAddress) {
  allocToSch[allocAddress].reserve(getAlloc(allocAddress).nOps());
  allocToSch[allocAddress].clear();
  for (OpAddress opAddress : getAlloc(allocAddress).getOps()) {
    allocToSch[allocAddress].push_back(opToSch[opAddress]);
  }
  std::sort(allocToSch[allocAddress].begin(), allocToSch[allocAddress].end());
}

std::vector<AllocWeight> Graph::getBwdRippleCosts(ScheduleIndex start0,
                                                  int nToShift,
                                                  int lastExtProd) const {

  const int sign             = -1;
  const auto nCostsToCompute = start0 - lastExtProd - 1;
  const auto dirOffset       = 0;
  return getRippleCosts(start0, nToShift, sign, nCostsToCompute, dirOffset);
}

// this was the trickiest function to get right
std::vector<AllocWeight> Graph::getRippleCosts(ScheduleIndex start0,
                                               int nToShift,
                                               int sign,
                                               int nCostsToCompute,
                                               int dirOffset) const {

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
    auto start1Allocs = scheduleToAllocs(start1 + dirOffset);
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
      ScheduleIndex extremum;
      if (sign == -1) {
        extremum = schedInds[0];
      } else {
        extremum = schedInds.back();
      }

      // only in a special case will incrWeight be non-zero:
      // TODO(T14829) diagram explaining this special case. The diagram exists
      // (jn) but is not publicly available. Sharepoint does not seem like a
      // good place to keep it
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

AllocWeight Graph::getMaxLiveness() const {
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

AllocWeight Graph::getSumLiveness() const {
  if (schToLiveness.empty()) {
    throw error(
        "Call go getSumLiveness, but schToLiveness has not yet been set");
  }
  return std::accumulate(schToLiveness.cbegin(),
                         schToLiveness.cend(),
                         static_cast<AllocWeight>(0),
                         [](AllocWeight a, AllocWeight b) { return a + b; });
}

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
