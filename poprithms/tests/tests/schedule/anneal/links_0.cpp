// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <vector>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {
using namespace poprithms::schedule::anneal;

void test0() {
  Graph g;
  auto rootOp = g.insertOp("root");
  auto tailOp = g.insertOp("tail");

  uint64_t nChains      = 10;
  uint64_t chain0Length = 5;

  for (uint64_t i = 0; i < 10; ++i) {
    auto parent = rootOp;
    for (uint64_t j = 0; j < chain0Length + i; ++j) {
      auto op = g.insertOp(std::to_string(i) + "_" + std::to_string(j));
      g.insertConstraint(parent, op);
      if (parent == rootOp) {
        g.insertConstraint(parent, op);
      } else {
        g.insertLink(parent, op);
      }
      parent = op;
    }
    g.insertConstraint(parent, tailOp);
  }

  auto linkMerged            = g.getLinkMerged();
  const auto &childGraph     = std::get<0>(linkMerged);
  const auto &parentGraphOps = std::get<1>(linkMerged);
  if (childGraph.nOps() != 1 + nChains + 1) {
    throw error("Expected each of the chains to have collapsed into 1 Op");
  }

  for (auto parentOps : parentGraphOps) {
    std::sort(parentOps.begin(), parentOps.end());
    for (uint64_t i = 1; i < parentOps.size(); ++i) {
      if (parentOps[i] != 1 + parentOps[i - 1]) {
        throw error(
            "expected the OpAddresses in each chain to be contiguous");
      }
    }
  }
}

void test1() {
  Graph g;
  auto ops = g.insertOps({"0", "1", "2", "3", "4"});
  g.insertLink(ops[0], ops[1]);
  g.insertLink(ops[2], ops[3]);
  g.insertLink(ops[3], ops[4]);
  g.insertConstraint(ops[4], ops[0]);
  if (!g.hasAtLeastOneLink()) {
    throw error("g should have at least q link: it should have 2");
  }
  auto chains = g.getLinkChains();
  if (chains.size() != 2) {
    throw error("There should be 2 chains, {1,2} and {3,4,5}");
  }
  std::sort(chains.begin(), chains.end());
  if (chains[0] != std::vector<OpAddress>{0, 1}) {
    throw error("Expected first chain to have addresses {0,1}");
  }
  if (chains[1] != std::vector<OpAddress>{2, 3, 4}) {
    throw error("Expected first chain to have addresses {2,3,4}");
  }
  g.initialize(KahnTieBreaker::GREEDY, 1011);
  if (g.getScheduleToOp() != std::vector<OpAddress>{2, 3, 4, 0, 1}) {
    throw error("Expected a different final schedule in test1");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
