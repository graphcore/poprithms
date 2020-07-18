// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <testutil/memory/nest/randomregion.hpp>

int main() {

  using namespace poprithms::memory::nest;

  poprithms::logging::enableDeltaTime(true);
  poprithms::logging::Logger loo("loo");

  const int nRuns                = 25;
  const int64_t nDistinctFactors = 3;
  int seed                       = 1011;

  for (int64_t nFactors : {5, 11}) {
    for (int64_t rank0 : {1, 2, 3}) {
      for (int64_t rank1 : {1, 2, 3}) {
        for (int64_t maxSettDepth : {0, 1, 3}) {
          for (uint64_t i = 0; i < nRuns; ++i) {

            ++seed;

            const auto x0 =
                getShapes(seed, rank0, rank1, nDistinctFactors, nFactors);

            const auto from = std::get<0>(x0);
            const auto to   = std::get<1>(x0);
            std::ostringstream oss;
            oss << i << ".  " << from << " -> " << to;
            loo.trace(oss.str());

            const auto fromRegion = getRandomRegion(
                from, /* seed */ 100 + i, /* max Sett depth */ 2);

            std::cout << "seed=" << seed << ", maxSettDepth=" << maxSettDepth
                      << ", nFactors=" << nFactors
                      << ", fromRegion = " << fromRegion << " to = " << to
                      << std::endl;
            const auto reshaped = fromRegion.reshape(to);
            std::vector<Region> returned;
            for (const auto &x : reshaped.get()) {
              auto y = x.reshape(from);
              returned.insert(
                  returned.end(), y.get().cbegin(), y.get().cend());
            }
            if (!Region::equivalent(DisjointRegions{fromRegion},
                                    DisjointRegions(from, returned))) {
              throw error("Darn");
            }
          }
        }
      }
    }
  }

  return 0;
}
