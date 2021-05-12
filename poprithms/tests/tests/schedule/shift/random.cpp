// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/opalloc.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <testutil/schedule/shift/randomgraph.hpp>
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

// N Ops,
// [1....E] producers for each Op randomly from D most previous
// each Op creates 1 new alloc, used allocs of all producers
// allocs have size in [10, 20)
//

int main(int argc, char **argv) {

  // N 40 E 5 D 20 graphSeed 1012 seed 114 : final sum is 5260
  // N 40 E 5 D 20 graphSeed 1012 seed 115 : final sum is 5242
  //
  // interestingly, for many different seeds, the final sum is always either
  // 5260 or 5242.

  using namespace poprithms::schedule::shift;

  auto opts = ShiftCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"N", "E", "D", "graphSeed"},
      {"Number of Ops",
       "Number of producers per Op",
       "range depth in past from which to select producers, randomly",
       "random source for selecting producers"});

  auto N         = std::stoi(opts.at("N"));
  auto E         = std::stoi(opts.at("E"));
  auto D         = std::stoi(opts.at("D"));
  auto graphSeed = std::stoi(opts.at("graphSeed"));

  auto opts1 = ShiftCommandLineOptions().getAlgoCommandLineOptionsMap(opts);
  opts1.insert({"kahnTieBreaker", "Random"});
  opts1.insert({"seed", "1015"});

  auto g        = getRandomGraph(N, E, D, graphSeed);
  const auto sg = ScheduledGraph(std::move(g), opts1);

  // nothing specific to test, we'll verify the sum liveness;
  std::vector<std::vector<ScheduleIndex>> allocToSched(sg.nAllocs());
  for (ScheduleIndex i = 0; i < sg.nOps_i32(); ++i) {
    OpAddress opAdd = sg.scheduleToOp(i);
    for (AllocAddress a : sg.getOp(opAdd).getAllocs()) {
      allocToSched[a].push_back(i);
    }
  }
  AllocWeight s{0};
  for (const auto &alloc : sg.getGraph().getAllocs()) {
    auto allocAddress = alloc.getAddress();
    if (!allocToSched[allocAddress].empty()) {
      auto cnt = (allocToSched[allocAddress].back() -
                  allocToSched[allocAddress][0] + 1);
      s += alloc.getWeight() * cnt;
    }
  }

  std::cout << sg.getLivenessString() << std::endl;

  if (s != sg.getSumLiveness()) {
    std::cout << s << " != " << sg.getSumLiveness() << std::endl;
    throw poprithms::schedule::shift::error(
        "Computed sum of final liveness incorrect in random example test");
  }

  if (sg.getGraph().getSerializationString() !=
          Graph::fromSerializationString(
              sg.getGraph().getSerializationString())
              .getSerializationString() ||
      sg.getGraph() != Graph::fromSerializationString(
                           sg.getGraph().getSerializationString())) {
    std::ostringstream oss;
    oss << "g.serialization != g.serialization(fromSerial(g.serialization)). "
        << "This suggests a problem with Graph serialization. ";
    oss << "The serialization of G is "
        << sg.getGraph().getSerializationString();
    throw error(oss.str());
  }
  return 0;
}
