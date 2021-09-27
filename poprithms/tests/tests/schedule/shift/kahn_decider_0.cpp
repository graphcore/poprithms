// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/fromcache.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::schedule::shift;

template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &ts) {
  poprithms::util::append(ost, ts);
  return ost;
}

template <typename T>
void assertSchedule(const std::vector<T> &expected,
                    const std::vector<T> &observed,
                    std::string ctxt) {
  if (expected != observed) {
    std::ostringstream oss;
    oss << "Expected to observe the schedule " << expected
        << ", but observed the schedule " << observed
        << " This for scheduler : " << ctxt;
    throw poprithms::test::error(oss.str());
  }
}

void test0() {

  //             +---^--+
  //             |      |
  //             |      |
  //             |      |
  //   root -->--+      +--->--- tail
  //             |      |
  //             +-->---+
  //             :      :
  //             |      |
  //             |      |
  //             +--->--+

  Graph g0;
  auto root = g0.insertOp("root");
  auto tail = g0.insertOp("tail");

  std::vector<OpAddress> opEdges;
  for (uint64_t i = 0; i < 10; ++i) {
    auto nxt   = g0.insertOp("edge" + std::to_string(i));
    auto alloc = g0.insertAlloc(100. - std::abs(nxt - 5.1));
    opEdges.push_back(nxt);
    g0.insertConstraint(root, nxt);
    g0.insertConstraint(nxt, tail);
    g0.insertOpAlloc({root, nxt}, alloc);
  }

  auto schedGreedy = ScheduledGraph(Graph(g0),
                                    {KahnTieBreaker::GREEDY, {}},
                                    TransitiveClosureOptimizations::allOff(),
                                    RotationTermination::preStart());
  assertSchedule({0, 5, 6, 4, 7, 3, 8, 2, 9, 10, 11, 1},
                 schedGreedy.viewInternalScheduleToOp(),
                 "Greedy.");

  auto schedFilo = ScheduledGraph(Graph(g0),
                                  {KahnTieBreaker::FIFO, {}},
                                  TransitiveClosureOptimizations::allOff(),
                                  RotationTermination::preStart());
  assertSchedule({0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
                 schedFilo.viewInternalScheduleToOp(),
                 "Filo.");

  auto schedRandom = ScheduledGraph(Graph(g0),
                                    {KahnTieBreaker::RANDOM, {}},
                                    TransitiveClosureOptimizations::allOff(),
                                    RotationTermination::preStart(),
                                    RotationAlgo::RIPPLE,
                                    1011);

  auto rando1 = schedRandom.viewInternalScheduleToOp();

  auto rando2 = ScheduledGraph(Graph(g0),
                               {KahnTieBreaker::RANDOM,
                                {{2, 10.}, {4, 9.}, {6, 8.}, {8, 7.}}},
                               TransitiveClosureOptimizations::allOff(),
                               RotationTermination::preStart(),
                               RotationAlgo::RIPPLE,
                               1011)
                    .viewInternalScheduleToOp();

  // Expect 0, 2, 4, 6, 8, ....

  assertSchedule<OpAddress>({0, 2, 4, 6, 8},
                            {rando2.cbegin(), rando2.cbegin() + 5},
                            "Random with warm start");
}

void test1() {

  uint32_t seed = 1011;
  std::mt19937 rng(seed);

  uint64_t nOps{100};
  uint64_t nAllocs{20};
  auto g = getRandomGraph(nOps, 2, 20, 1011);
  for (uint64_t i = 0; i < nAllocs; ++i) {
    auto a = g.insertAlloc(10.);
    for (uint64_t j = 0; j < 5; ++j) {
      g.insertOpAlloc(rng() % nOps, a);
    }
  }
  auto oracle = ScheduledGraph(Graph(g),
                               KahnDecider(KahnTieBreaker::GREEDY),
                               TransitiveClosureOptimizations::allOn(),
                               RotationTermination::nHours(1),
                               RotationAlgo::RIPPLE);

  // The fraction of Ops which get a hint from the exact solution:
  // We expect higher fraction => less swapping to get solution.
  std::vector<double> fracsFixed{0.0, 0.5, 1.0};

  std::vector<uint64_t> nRotations;
  for (auto fFixed : fracsFixed) {
    KahnDecider::Priorities pris;
    for (uint64_t i = 0; i < g.nOps(); ++i) {
      // if (foo.opToSchedule(i) < fFixed * g.nOps()) {
      if (i < fFixed * g.nOps()) {
        pris.push_back(
            {OpAddress(i),
             static_cast<double>(g.nOps() + 100. - oracle.opToSchedule(i))});
      }
    }

    SwitchSummaryWriter ig;

    auto foo2 = fromCache(Graph(g),
                          Settings(KahnDecider(KahnTieBreaker::RANDOM, pris),
                                   TransitiveClosureOptimizations::allOn(),
                                   RotationTermination::nHours(1)),
                          ig);

    nRotations.push_back(ig.allChanges().size());
  }

  for (uint64_t i = 0; i < fracsFixed.size(); ++i) {
    std::cout << fracsFixed[i] << " : " << nRotations[i] << std::endl;
  }

  for (uint64_t i = 1; i < fracsFixed.size(); ++i) {
    if (nRotations[i - 1] <= nRotations[i]) {
      throw poprithms::test::error("Expect lower fraction of Ops with hints "
                                   "to result in more swaps. ");
    }
  }
  if (nRotations.back() != 0) {
    throw poprithms::test::error(
        "Expect no swaps when all Ops have hint from exact solution. ");
  }
}
} // namespace

int main() {
  test0();
  test1();
  return 0;
}
