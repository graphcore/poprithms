// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

poprithms::schedule::shift::Graph getRandomGraph(uint64_t N,
                                                 uint64_t E,
                                                 uint64_t D,
                                                 int32_t graphSeed,
                                                 uint64_t lowAlloc,
                                                 uint64_t highAlloc) {

  std::mt19937 gen(graphSeed);

  std::vector<int> dBack(D);
  // -D ... -1
  std::iota(dBack.begin(), dBack.end(), -D);

  Graph g;
  for (int n = 0; n < N; ++n) {

    auto n_u64 = static_cast<uint64_t>(n);
    if (n < D) {
      g.insertOp("op_" + std::to_string(n));
    } else {
      std::vector<int> samples;
      samples.reserve(E);
      std::sample(
          dBack.begin(), dBack.end(), std::back_inserter(samples), E, gen);
      for (auto &x : samples) {
        x += n;
      }
      std::vector<OpAddress> prods;
      for (auto x : samples) {
        auto x_u64 = static_cast<OpAddress>(x);
        prods.push_back(x_u64);
      }
      g.insertOp(
          prods, std::vector<AllocAddress>{}, "op_" + std::to_string(n));
    }
  }
  addConnectedAllocs(g, lowAlloc, highAlloc, graphSeed);
  return g;
}

void addConnectedAllocs(Graph &g,
                        uint64_t lowAlloc,
                        uint64_t highAlloc,
                        uint32_t seed) {

  if (highAlloc <= lowAlloc) {
    std::ostringstream oss;
    oss << "Expect highAlloc > lowAlloc in addConnectedAllocs, but highAlloc="
        << highAlloc << " and lowAlloc = " << lowAlloc;
    throw poprithms::test::error(oss.str());
  }

  std::mt19937 rng(seed);
  for (uint64_t i = 0; i < g.nOps(); ++i) {
    auto w = g.insertAlloc(lowAlloc + rng() % (highAlloc - lowAlloc));
    g.insertOpAlloc(i, w);
    for (auto o : g.getOp(i).getOuts()) {
      g.insertOpAlloc(o, w);
    }
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
