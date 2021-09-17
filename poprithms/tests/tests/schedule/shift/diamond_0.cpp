// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <string>

#include <testutil/schedule/shift/diamond_generator.hpp>
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::shift;
  auto opts = ShiftCommandLineOptions().getCommandLineOptionsMap(
      argc, argv, {"N"}, {"The number of intermediate Ops in the diamond"});
  auto N      = std::stoull(opts.at("N"));
  Graph graph = getDiamondGraph0(N);
  ScheduledGraph sg(std::move(graph),
                    {KahnTieBreaker::RANDOM, {}},
                    TransitiveClosureOptimizations::allOn());
  assertGlobalMinimumDiamondGraph0(sg, N);
}
