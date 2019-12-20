#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <testutil/schedule/anneal/commandlineoptions.hpp>
#include <tuple>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>

// N Ops,
// [1....E] producers for each Op randomly from D most previous
// each Op creates 1 new alloc, used allocs of all producers
// allocs have size in [10, 20)
//

poprithms::schedule::anneal::Graph
getGraph(uint64_t N, uint64_t E, uint64_t D, int graphSeed) {

  using namespace poprithms::schedule::anneal;

  std::mt19937 gen(graphSeed);
  std::uniform_int_distribution<> distSizeAlloc(10, 19);

  std::vector<int> dBack(D);
  // -D ... -1
  std::iota(dBack.begin(), dBack.end(), -D);

  Graph g;

  for (int n = 0; n < N; ++n) {
    auto allocId = g.insertAlloc(distSizeAlloc(gen));
  }

  for (int n = 0; n < N; ++n) {
    auto n_u64 = static_cast<uint64_t>(n);
    if (n < D) {
      g.insertOp({}, {n_u64}, "op_" + std::to_string(n));
    } else {
      std::vector<int> samples;
      samples.reserve(E);
      std::sample(
          dBack.begin(), dBack.end(), std::back_inserter(samples), E, gen);
      for (auto &x : samples) {
        x += n;
      }
      std::vector<OpAddress> prods;
      std::vector<AllocAddress> allocs{n_u64};
      for (auto x : samples) {
        auto x_u64 = static_cast<uint64_t>(x);
        prods.push_back(x_u64);
        allocs.push_back(x_u64);
      }
      g.insertOp(prods, allocs, "op_" + std::to_string(n));
    }
  }
  return g;
}

int main(int argc, char **argv) {

  // N 40 E 5 D 20 graphSeed 1012 seed 114 : final sum is 5260
  // N 40 E 5 D 20 graphSeed 1012 seed 115 : final sum is 5242
  //
  // interestingly, for many different seeds, the final sum is always either
  // 5260 or 5242.

  using namespace poprithms::schedule::anneal;

  auto opts = CommandLineOptions::getCommandLineOptionsMap(
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

  auto g = getGraph(N, E, D, graphSeed);
  g.initialize(KhanTieBreaker::RANDOM, 1015);
  g.minSumLivenessAnneal(
      CommandLineOptions::getAnnealCommandLineOptionsMap(opts));

  // nothing specific to test, we'll verify the sum liveness;
  std::vector<std::vector<ScheduleIndex>> allocToSched(g.nAllocs());
  for (ScheduleIndex i = 0; i < g.nOps_i32(); ++i) {
    OpAddress a = g.scheduleToOp(i);
    for (AllocAddress a : g.getOp(a).getAllocs()) {
      allocToSched[a].push_back(i);
    }
  }
  AllocWeight s{0};
  for (const auto &alloc : g.getAllocs()) {
    auto allocAddress = alloc.getAddress();
    if (!allocToSched[allocAddress].empty()) {
      s += alloc.getWeight() * (allocToSched[allocAddress].back() -
                                allocToSched[allocAddress][0] + 1);
    }
  }

  std::cout << g.getLivenessString() << std::endl;

  if (s != g.getSumLiveness()) {
    std::cout << s << " != " << g.getSumLiveness() << std::endl;
    throw poprithms::error(
        "Computed sum of final liveness incorrect in random example test");
  }
  return 0;
}
