// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <testutil/schedule/shift/randomgraph.hpp>

// In this test, we check that allowing more swaps results in lower schedule
// livenesses. We only test this swapLimitCount, a test for timeLimitSeconds
// will be flakey. timeLimitSeconds has been tested manually for now.

int main() {

  using namespace poprithms::schedule::shift;

  const int N         = 100;
  const int E         = 10;
  const int D         = 40;
  const int graphSeed = 1011;
  const std::vector<int64_t> swapLimitCounts{-100, 1, 1000};
  std::vector<AllocWeight> livenesses;

  for (auto swapLimitCount : swapLimitCounts) {

    auto g = getRandomGraph(N, E, D, graphSeed);

    uint32_t seed           = 1012;
    double timeLimitSeconds = 1000.0;

    ScheduledGraph sg(std::move(g),
                      KahnTieBreaker::RANDOM,
                      Settings::defaultTCOs(),
                      {timeLimitSeconds, swapLimitCount},
                      RotationAlgo::RIPPLE,
                      seed,
                      DebugMode::On);

    livenesses.push_back(sg.getSumLiveness());
  }

  std::cout << "Livenesses at progressively increasing swap count limits: "
            << std::endl;
  for (auto x : livenesses) {
    std::cout << x << std::endl;
  }

  for (auto x = std::next(livenesses.cbegin()); x != livenesses.cend();
       std::advance(x, 1)) {
    auto atLowerLimit = std::prev(x);
    if (!(*x < *atLowerLimit)) {
      throw poprithms::test::error(
          "Expected liveness to be lower at higher swap limit");
    }
  }
  return 0;
}
