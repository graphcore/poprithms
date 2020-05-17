#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/transitiveclosureoptimizations.hpp>

void test0() {
  using namespace poprithms::schedule::anneal;
  Graph g;

  // clang-format off
 std::vector<double> allocWeights = 
                          {6,   5,   3,   5,   7,   3,   6};
  auto ops = g.insertOps({"0", "1", "2", "3", "4", "5", "6", "7"});
  //                       ------------------------------------
  //                      +2   +3   +4   -6   -2   +4   +5   -9
  //                      ==   ==   =================   =======
  //
  // clang-format on

  for (uint64_t i = 1; i < g.nOps(); ++i) {
    g.insertConstraint(ops[i - 1], ops[i]);
    auto alloc = g.insertAlloc(allocWeights[i - 1]);
    g.insertOpAlloc({ops[i - 1], ops[i]}, alloc);
  }

  auto gCopy = g;
  gCopy.initialize(
      KahnTieBreaker::RANDOM, 1011, TransitiveClosureOptimizations::allOff());
  if (!gCopy.getLinkChains().empty()) {
    throw error("With all transitive closure optimizations off, expected no "
                "link chains");
  }

  const auto pom =
      TransitiveClosureOptimizations::allOff().withLinkTightDrops();
  g.initialize(KahnTieBreaker::RANDOM, 1011, pom);

  auto chainLinks = g.getLinkChains();

  if (chainLinks.size() != 3) {
    throw error("Expected 3 chains");
  }

  if (chainLinks[0] != std::vector<OpAddress>{0, 1, 2} ||
      chainLinks[1] != std::vector<OpAddress>{3, 4, 5} ||
      chainLinks[2] != std::vector<OpAddress>{6, 7}) {
    throw error("Chain links not as expected in test0");
  }
}

void test1() {
  // a larger, random version of test0.

  using namespace poprithms::schedule::anneal;
  uint64_t nOps = 60;

  std::mt19937 gen(1015);
  // values drawn from [nOps, 2*nOps)
  std::uniform_int_distribution<> distSizeAlloc(nOps, 2 * nOps);
  auto getNextVal = [&gen, &distSizeAlloc]() { return distSizeAlloc(gen); };
  Graph g;
  std::vector<OpAddress> ops;
  std::vector<double> allocVals;
  for (uint64_t i = 0; i < nOps; ++i) {
    ops.push_back(g.insertOp("Op" + std::to_string(i)));
    if (i > 0) {
      auto id0 = *std::next(ops.crbegin());
      auto id1 = *(ops.crbegin());
      g.insertConstraint(id0, id1);
      auto wVal  = getNextVal() - i;
      auto alloc = g.insertAlloc(wVal);
      allocVals.push_back(wVal);
      g.insertOpAlloc({id0, id1}, alloc);
    }
  }
  std::vector<double> allocDeltas{allocVals[0]};
  for (uint64_t i = 1; i < nOps - 1; ++i) {
    allocDeltas.push_back(allocVals[i] - allocVals[i - 1]);
  }
  allocDeltas.push_back(-1. * allocVals.back());

  std::vector<std::vector<OpAddress>> expectedChains;
  for (uint64_t index = 0; index < nOps; ++index) {
    if (!expectedChains.empty() &&
        allocDeltas[index] <= allocDeltas[index - 1]) {
      expectedChains.back().push_back(index);
    } else if (index != nOps - 1 &&
               allocDeltas[index] >= allocDeltas[index + 1]) {
      expectedChains.push_back({index});
    }
  }

  const auto pom =
      TransitiveClosureOptimizations::allOff().withLinkTightDrops();
  g.initialize(KahnTieBreaker::RANDOM, 1011, pom);
  auto chainLinks = g.getLinkChains();

  bool printExpectedAndObserved = false;
  if (printExpectedAndObserved) {
    std::cout << "\nexpected chains ids: " << std::endl;
    for (auto x : expectedChains) {
      for (auto y : x) {
        std::cout << "   " << y;
      }
      std::cout << std::endl;
    }
    std::cout << "\nexpected chain deltas: " << std::endl;
    for (auto x : expectedChains) {
      for (auto y : x) {
        std::cout << "   " << allocDeltas[y];
      }
      std::cout << std::endl;
    }
    std::cout << "\nobserved chains : " << std::endl;
    for (auto x : chainLinks) {
      for (auto y : x) {
        std::cout << "  " << y;
      }
      std::cout << std::endl;
    }
  }

  if (chainLinks.size() != expectedChains.size()) {
    throw error("number of chains not as expected");
  }

  if (chainLinks != expectedChains) {
    throw error("Chain links not as expected");
  }
}

int main() {
  test0();
  test1();
  return 0;
}
