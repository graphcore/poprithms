// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <poprithms/schedule/shift/scheduledgraph.hpp>

using namespace poprithms::schedule::shift;

class OpAlloc {
public:
  OpAlloc(OpAddress o, AllocAddress a) : op(o), alloc(a) {}
  OpAddress op;
  AllocAddress alloc;
};

int main() {

  // N Ops,
  // E constraints,
  // K allocs with number of Ops random from
  //   [1...P] each

  auto getGraph = [](uint64_t N, uint64_t E, uint64_t K, int P, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> opDis(0, N - 1);
    std::uniform_int_distribution<> allocCountDis(1, P);

    // generate 1 non-self edge
    auto getEdge = [&opDis, &gen]() {
      auto a = opDis(gen);
      auto b = a;
      while (b == a) {
        b = opDis(gen);
      }
      if (b < a) {
        std::swap(a, b);
      }
      return std::tuple(a, b);
    };

    // generate E unique non-self edges
    std::vector<std::tuple<int, int>> edges;
    while (edges.size() < E) {
      auto edge = getEdge();
      while (std::find(edges.cbegin(), edges.cend(), edge) != edges.cend()) {
        edge = getEdge();
      }
      edges.push_back(edge);
    }

    // allocations
    std::vector<std::vector<int>> allocsToOps;
    allocsToOps.reserve(K);
    while (allocsToOps.size() < K) {
      allocsToOps.push_back({});
      auto nAllocOps = allocCountDis(gen);
      allocsToOps.back().reserve(nAllocOps);
      for (int a = 0; a < nAllocOps; ++a) {
        auto op                = opDis(gen);
        const auto &allocToOps = allocsToOps.back();
        while (std::find(allocToOps.cbegin(), allocToOps.cend(), op) !=
               allocToOps.cend()) {
          op = opDis(gen);
        }
        allocsToOps.back().push_back(op);
      }
    }

    Graph g;
    for (uint64_t k = 0; k < K; ++k) {
      g.insertAlloc(1);
    }

    for (uint64_t n = 0; n < N; ++n) {
      std::vector<OpAddress> producers;
      for (const auto &edge : edges) {
        if (static_cast<uint64_t>(std::get<1>(edge)) == n) {
          producers.push_back(static_cast<OpAddress>(std::get<0>(edge)));
        }
      }
      std::vector<AllocAddress> opToAllocs;
      AllocAddress allocAddress{0};
      for (const auto &allocToOp : allocsToOps) {
        if (std::find(allocToOp.cbegin(), allocToOp.cend(), n) !=
            allocToOp.cend()) {
          opToAllocs.push_back(allocAddress);
        }
        ++allocAddress;
      }
      g.insertOp(producers, opToAllocs, "op_" + std::to_string(n));
    }

    return g;
  };

  std::random_device r;
  int seed   = r();
  int nTests = 0;

  while (nTests < 10) {

    ++nTests;

    ++seed;
    std::cout << "\nRandom test with seed = " << seed << std::endl;

    uint64_t nOps             = 40;
    uint64_t nEdges           = 40;
    uint64_t nAllocs          = 60;
    uint64_t opsPerAllocUpper = 5;

    auto g = getGraph(nOps, nEdges, nAllocs, opsPerAllocUpper, seed);

    auto g0 = g;

    ScheduledGraph sg0(std::move(g0),
                       {KahnTieBreaker::RANDOM, {}},
                       TransitiveClosureOptimizations::allOn(),
                       RotationTermination::preStart());

    std::vector<AllocWeight> lBefore;
    auto totalBefore = AllocWeight::zero();
    auto maxBefore   = AllocWeight::zero();
    for (uint64_t i = 0; i < nOps; ++i) {
      auto x = sg0.scheduleToLiveness(i);
      totalBefore += x;
      maxBefore = std::max(maxBefore, x);
    }

    ScheduledGraph sg(std::move(g),
                      {KahnTieBreaker::RANDOM, {}},
                      TransitiveClosureOptimizations::allOn(),
                      RotationTermination::nHours(10),
                      RotationAlgo::RIPPLE,
                      1011,
                      FileWriter::None(),
                      DebugMode::On);

    std::vector<AllocWeight> lAfter;
    AllocWeight maxAfter   = AllocWeight::zero();
    AllocWeight totalAfter = AllocWeight::zero();
    for (uint64_t i = 0; i < nOps; ++i) {
      auto x = sg.scheduleToLiveness(i);
      totalAfter += x;
      maxAfter = std::max(maxAfter, x);
    }
    std::cout << "max   : " << maxBefore << " ---> " << maxAfter << std::endl;
    std::cout << "total : " << totalBefore << " ---> " << totalAfter
              << std::endl;
  }

  return 0;
}
