// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/opalloc.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <testutil/schedule/shift/bifurcate_generator.hpp>
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::shift;
  auto opts = ShiftCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"D"},
      {"The depth of the bifurcating-merging graph, The number of nodes "
       "grows as 2**D"});

  uint64_t D = std::stoi(opts.at("D"));
  auto annos = ShiftCommandLineOptions().getAlgoCommandLineOptionsMap(opts);

  auto g  = getBifurcatingGraph0(D);
  auto sg = ScheduledGraph(std::move(g), annos);

  assertGlobalMinimumBifurcatingGraph0(sg, D);
  return 0;
}
