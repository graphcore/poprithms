// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/schedule/shift/scheduledgraph.hpp>

int main() {
  using namespace poprithms::schedule::shift;
  // Verify that Graphs with 0, 1 and 2 Ops are constructed and scheduled
  // without failure
  for (int i = 0; i < 3; ++i) {
    Graph graph;
    for (int j = 0; j < i; ++j) {
      graph.insertOp("op" + std::to_string(10 * i + j));
    }
    ScheduledGraph sg(std::move(graph),
                      KahnTieBreaker::RANDOM,
                      TransitiveClosureOptimizations::allOn(),
                      RotationTermination::nHours(10),
                      RotationAlgo::RIPPLE,
                      1011,
                      FileWriter::None(),
                      DebugMode::On);
  }
  return 0;
}
