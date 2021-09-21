// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <fstream>
#include <ostream>
#include <random>

#include <testutil/schedule/base/randomdag.hpp>
#include <testutil/schedule/shift/bifurcate_generator.hpp>
#include <testutil/schedule/shift/diamond_generator.hpp>
#include <testutil/schedule/shift/grid_generator.hpp>
#include <testutil/schedule/shift/randomgraph.hpp>
#include <testutil/schedule/shift/recompute_generator.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/schedulechange.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::schedule::shift;
using namespace poprithms::schedule;

constexpr uint64_t allocLow{4};
constexpr uint64_t allocHigh{7};
constexpr uint32_t seed{1011};

Graph getAdversary() {

  std::mt19937 rng(seed);

  // create a random DAG:
  const auto edges =
      poprithms::schedule::baseutil::randomConnectedDag(30, 1011);

  // randonly schedule the DAG:
  const auto genieSchedule =
      vanilla::Scheduler<uint64_t, double>::random(edges,
                                                   {}, // no priorities
                                                   {}, // no links
                                                   seed + 100,
                                                   vanilla::ErrorIfCycle::Yes,
                                                   vanilla::VerifyEdges::Yes);
  Graph g(edges);

  // Add allocations to DAG, which ensure that the random schedule above is
  // the optimal schedule. Do this by adding allocations which only span
  // contiguous regions of the schedule, so that for every allocation the
  // random schedule above is the optimal schedule.
  for (uint64_t i = 0; i < genieSchedule.size(); ++i) {
    if (i > 1) {
      auto a = g.insertAlloc(2 + rng() % 4);

      // 3 contiguous ops get this allocation.
      g.insertOpAlloc(
          {genieSchedule[i - 2], genieSchedule[i - 1], genieSchedule[i]}, a);
    }
  }

  return g;
}

Graph getTree() {
  uint32_t edgesPerOp = 1;
  uint32_t history    = 7;
  auto g = getRandomGraph(50, edgesPerOp, history, seed, allocLow, allocHigh);
  return g;
}

Graph getRecompute() {
  auto g = getRecomputeGraph(getLogNSeries(17), allocLow, allocHigh);
  return g;
}

Graph getGrid() { return getGridGraph0(8, allocLow, allocHigh, seed); }

Graph getBifurcating() {
  return getBifurcatingGraph0(5, allocLow, allocHigh, seed);
}

void process(Graph g, const std::string &dataWriteDir) {

  SwitchSummaryWriter sww;

  // Schedule the graph 'g', logging information to the SwitchSummaryWriter
  // 'sww'.
  auto sg0 = ScheduledGraph(Graph(g),
                            KahnDecider(KahnTieBreaker::RANDOM),
                            TransitiveClosureOptimizations::allOff(),
                            RotationTermination::nHours(1),
                            RotationAlgo::RIPPLE,
                            1011,
                            sww);

  // The directory where 'sww' will write log files. We write this directory
  // name to the file 'dateWriteDir.txt', which is used in change_reader_0.py
  // to locate the log files.
  std::ofstream out("dataWriteDir.txt");
  if (!out.is_open()) {
    throw poprithms::test::error(
        "Failed to open file, destined to contain the name of the file "
        "which the SwitchSummaryWriter writes to. ");
  }
  out << dataWriteDir;

  sww.writeToFile(dataWriteDir);
}

} // namespace

int main() {

  enum class Mode { Test = 0, GenerateAll };
  auto mode = Mode::Test;

  switch (mode) {
  case Mode::Test: {
    process(getRecompute(), "tempChangesWriteReadTests");
    break;
  }
  case Mode::GenerateAll: {
    process(getRecompute(), "recomputeLog");
    process(getTree(), "treeLog");
    process(getGrid(), "gridLog");
    process(getBifurcating(), "bifurcatingLog");
    process(getAdversary(), "adversaryLog");
    break;
  }
  }

  return 0;
}
