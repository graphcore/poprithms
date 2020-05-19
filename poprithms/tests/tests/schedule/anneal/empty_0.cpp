// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/schedule/anneal/graph.hpp>

int main() {
  using namespace poprithms::schedule::anneal;
  // Verify that Graphs with 0, 1 and 2 Ops are constructed and scheduled
  // without failure
  for (int i = 0; i < 3; ++i) {
    Graph graph;
    for (int j = 0; j < i; ++j) {
      graph.insertOp("op" + std::to_string(10 * i + j));
    }
    std::cout << "initialize" << std::endl;
    graph.initialize(KahnTieBreaker::RANDOM, 1011);
    std::cout << "minSumLiveness" << std::endl;
    graph.minSumLivenessAnneal({{"debug", "1"}});
  }
  return 0;
}
