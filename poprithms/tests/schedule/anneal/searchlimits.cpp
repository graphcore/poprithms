#include <iostream>
#include <testutil/schedule/anneal/randomgraph.hpp>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

// In this test, we check that allowing more swaps results in lower schedule
// livenesses. We only test this swapLimitCount, a test for timeLimitSeconds
// will be flakey. timeLimitSeconds has been tested manually for now.

int main() {

  using namespace poprithms::schedule::anneal;

  const int N         = 100;
  const int E         = 10;
  const int D         = 40;
  const int graphSeed = 1011;
  const std::vector<int64_t> swapLimitCounts{-100, 1, 1000};
  std::vector<AllocWeight> livenesses;

  for (auto swapLimitCount : swapLimitCounts) {

    auto g = getRandomGraph(N, E, D, graphSeed);

    g.initialize(KahnTieBreaker::RANDOM, 1015);

    bool debug              = false;
    uint32_t seed           = 1012;
    double pStayPut         = 10.0;
    double pHigherFallRate  = 0.0;
    double pClimb           = 0.0;
    bool logging            = true;
    double timeLimitSeconds = 1000.0;

    g.minSumLivenessAnneal(MinSumLivenessAlgo::RIPPLE,
                           debug,
                           seed,
                           pStayPut,
                           pHigherFallRate,
                           pClimb,
                           logging,
                           timeLimitSeconds,
                           swapLimitCount);

    livenesses.push_back(g.getSumLiveness());
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
      throw error("Expected liveness to be lower at higher swap limit");
    }
  }
  return 0;
}
