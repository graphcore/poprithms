// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/util/printiter.hpp>

/*

      0
     / \
    1   4--5--6
    |    \    |
    2     8   7
     \     \  |
      3     9 10
       \     \/
        11   12
         \   /
           13


 */

namespace {
poprithms::schedule::shift::ScheduledGraph
getGraph(const std::vector<std::vector<uint64_t>> &tiers) {

  constexpr uint64_t nOps{14};

  using namespace poprithms::schedule::shift;

  // verify validity of tiers:
  std::vector<uint64_t> flattened;
  flattened.reserve(nOps);
  for (const auto &x : tiers) {
    flattened.insert(flattened.end(), x.cbegin(), x.cend());
  }
  std::sort(flattened.begin(), flattened.end());
  std::vector<uint64_t> allIndices(nOps);
  std::iota(allIndices.begin(), allIndices.end(), 0);
  if (flattened != allIndices) {
    std::ostringstream oss;
    oss << "Expected indices in tiers to be integers {0"
        << "..." << nOps - 1 << "}, not ";
    poprithms::util::append(oss, flattened);
    oss << '.';
    throw error(oss.str());
  }

  Graph g;
  for (uint64_t i = 0; i < nOps; ++i) {
    g.insertOp("op" + std::to_string(i));
  }
  for (uint64_t i : {0, 1, 2, 4, 5, 6, 8}) {
    g.insertConstraint(i, i + 1);
  }
  g.insertConstraint(3, 11);
  g.insertConstraint(0, 4);
  g.insertConstraint(4, 8);
  g.insertConstraint(11, 13);
  g.insertConstraint(12, 13);
  g.insertConstraint(9, 12);
  g.insertConstraint(7, 10);
  g.insertConstraint(10, 12);

  double alloc0 = 20.0;
  for (const auto &tier : tiers) {
    for (auto id : tier) {
      auto alloc = g.insertAlloc(alloc0);
      g.insertOpAlloc({0, id}, alloc);
    }
    alloc0 -= 1.0;
  }

  ScheduledGraph sg(
      std::move(g),
      KahnTieBreaker::RANDOM,
      TransitiveClosureOptimizations::allOff().withLinkCloseTightPairs(),
      RotationTermination::preStart());

  return sg;
}

std::vector<std::vector<uint64_t>>
getLinkChains(const std::vector<std::vector<uint64_t>> &tiers) {
  auto g      = getGraph(tiers);
  auto chains = g.getGraph().getLinkChains();
  for (auto &chain : chains) {
    std::sort(chain.begin(), chain.end());
  }
  std::sort(chains.begin(), chains.end());
  return chains;
}

void test0() {

  /*

       0
      / \
     1   4--5--6
     |    \    |
     2     8   7
      \     \  |
       3     9 10
        \     \/
         11   12
          \   /
            13
  */

  using namespace poprithms::schedule::shift;
  using LChains = std::vector<std::vector<uint64_t>>;

  auto chains =
      getLinkChains({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}});
  if (chains != LChains{{1, 2, 3, 11}, {5, 6, 7, 10}, {8, 9}}) {
    throw error("Expected all chains in the case where all Ops have same "
                "liveness change. We tie-break on the side of linking");
  }

  chains = getLinkChains(
      {{0, 13}, {1}, {2}, {5}, {8}, {6}, {9}, {7}, {4, 12}, {10}, {3}, {11}});
  //            ========                      ===...........====  =========
  if (chains != LChains{{1, 2}, {3, 11}, {7, 10}}) {
    throw error("Expected 3 chains of 2 in this particular case");
  }

  chains = getLinkChains(
      {{1}, {11}, {3}, {0, 13}, {2}, {7, 10}, {5, 6}, {4}, {9}, {8}, {12}});
  //   ===============..........===  ==============        ========
  if (chains != LChains{{1, 2, 3, 11}, {5, 6, 7, 10}, {8, 9}}) {
    throw error("Expected 3 chains of 2 in this first case");
  }

  chains = getLinkChains(
      {{{1}, {10}, {8}, {2}, {9}, {3}, {12}, {11}, {5, 6, 7}, {0, 4, 13}}});
  if (chains != LChains{{5, 6, 7}}) {
    throw error("Expected just the 1 chain with no intersecting intruders to "
                "be linked");
  }

  chains = getLinkChains(
      {{9, 4}, {0, 8}, {3, 13}, {11, 2}, {7, 5}, {10, 1}, {6, 12}});
  if (chains != LChains{{2, 3, 11}, {7, 10}, {8, 9}}) {
    throw error("Unexpected L1 Chains in this test");
  }
}
} // namespace

int main() {

  test0();
  return 0;
}
