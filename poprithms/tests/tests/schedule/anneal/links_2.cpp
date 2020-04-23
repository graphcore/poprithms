#include <iostream>
#include <vector>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <testutil/schedule/anneal/randomgraph.hpp>

namespace {
using namespace poprithms::schedule::anneal;

void test0() {

  // The linked diamond,
  //
  //    X0
  //  //  \
  // X1    X2
  //  \  //
  //    X3
  //
  // but with allocs which would prefer the mirror-image linkage. Are the
  // links preserved?

  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[2], ops[3]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[1], ops[3]);

  // Allocs want to go against the links:
  auto alloc0 = g.insertAlloc(100.0f);
  auto alloc1 = g.insertAlloc(100.0f);
  g.insertOpAlloc({ops[0], ops[2]}, alloc0);
  g.insertOpAlloc({ops[1], ops[3]}, alloc0);

  g.initialize(KahnTieBreaker::RANDOM, 1011);
  g.minSumLivenessAnneal({});

  if (g.scheduleToOp(0) != 0 || g.scheduleToOp(1) != 1 ||
      g.scheduleToOp(2) != 2 || g.scheduleToOp(3) != 3) {
    throw error("The Links between Ops are not satisfied");
  }
}

void test1() {

  //
  // A random test that links are preserved
  //

  int seed0 = 1011;
  auto nOps = 120;
  auto g0   = getRandomGraph(nOps, 3, 7, seed0);
  auto g1   = g0;

  auto initializeGraph = [seed0](Graph &g) {
    g.initialize(KahnTieBreaker::RANDOM, seed0);
  };
  initializeGraph(g0);

  // g1 is like g0, but with a few Links inserted
  for (ScheduleIndex i = 0; i < nOps - 1; ++i) {
    if (i % 3 == 0) {
      g1.insertLink(g0.scheduleToOp(i), g0.scheduleToOp(i + 1));
    }
  }
  initializeGraph(g1);

  g0.minSumLivenessAnneal({});
  g1.minSumLivenessAnneal({{"debug", "1"}});

  // 1) confirm that Links are all satisified
  for (ScheduleIndex i = 0; i < nOps - 1; ++i) {
    const auto &op0 = g1.getOp(g1.scheduleToOp(i));
    if (op0.hasForwardLink()) {
      auto op1Address = op0.getForwardLink();
      if (g1.opToSchedule(op1Address) != i + 1) {
        throw error("Link is not satisfied");
      }
    }
  }

  auto linkLessSum = g0.getSumLiveness();
  auto linkedSum   = g1.getSumLiveness();
  std::cout << "Link-less energy : " << linkLessSum << std::endl;
  std::cout << "With-link energy : " << linkedSum << std::endl;
  if (linkLessSum >= linkedSum) {
    throw error("That is (very) odd, random links in a random graph result "
                "in a lower annealed liveness sum");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
