// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <numeric>
#include <string>

#include <testutil/schedule/shift/recompute_generator.hpp>
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/opalloc.hpp>

int main(int argc, char **argv) {

  using namespace poprithms;
  using namespace poprithms::schedule::shift;
  auto opts = ShiftCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"N", "type"},
      {"The number of forward Ops",
       "The type of recomputation. Either sqrt: checkpoints at "
       "approximately every root(N) interval, or log: multi-depth "
       "recursion, where at each depth just the mid-point is checkpoint, "
       "and there approximately log(N) depths "});
  uint64_t nFwd             = std::stoi(opts.at("N"));
  std::string recomputeType = opts.at("type");
  std::vector<int> pattern;
  if (recomputeType == "sqrt") {
    pattern = getSqrtSeries(nFwd);
  } else if (recomputeType == "log") {
    pattern = getLogNSeries(nFwd);
  } else {
    throw poprithms::test::error(
        "Invalid type, log and sqrt are the current options");
  }
  auto g = getRecomputeGraph(pattern);
  //  g.initialize();
  //  std::cout << g.getLivenessString() << std::endl;

  auto sg = ScheduledGraph(
      std::move(g),
      ShiftCommandLineOptions().getAlgoCommandLineOptionsMap(opts));

  std::cout << sg.getLivenessString() << std::endl;

  assertGlobalMinimumRecomputeGraph0(sg);

  return 0;
}
