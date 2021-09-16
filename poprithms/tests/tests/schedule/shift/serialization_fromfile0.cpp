// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <fstream>
#include <streambuf>
#include <string>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

int main() {

  // testgraph0.json.in in the test source directory is copied to
  // testgraph0.json in the build directory, so when this test is executed,
  // the .json file is in the same directory as the binary executable.
  std::ifstream t("testgraph0.json");

  std::stringstream buffer;
  buffer << t.rdbuf();
  auto s = buffer.str();

  using namespace poprithms::schedule::shift;
  auto g = Graph::fromSerializationString(s);

  ScheduledGraph sgoo(Graph(g),
                      KahnTieBreaker::FIFO,
                      TransitiveClosureOptimizations::allOn(),
                      RotationTermination::nHours(1),
                      RotationAlgo::RIPPLE,
                      1011,
                      FileWriter::None(),
                      DebugMode::Off);

  return 0;
}
