// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

int main() {

  std::ifstream t("testgraph0.json");
  // std::ifstream
  // t("/Users/jamesn/T44427_logs/time2__nOps27904__uid0/graph0.json");

  std::stringstream buffer;
  buffer << t.rdbuf();
  auto s = buffer.str();

  using namespace poprithms::schedule::shift;
  ScheduledGraph sgoo(Graph::fromSerializationString(s),
                      KahnTieBreaker::FIFO,
                      TransitiveClosureOptimizations::allOn(),
                      RotationTermination::nHours(1),
                      RotationAlgo::RIPPLE,
                      1011,
                      DebugMode::Off);

  return 0;
}
