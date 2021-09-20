// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>

#include <testutil/memory/nest/randomregion.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace nest {

std::array<Shape, 2> getShapes(uint32_t seed,
                               uint64_t l0,
                               uint64_t l1,
                               uint64_t nDistinctFactors,
                               uint64_t nFactors) {
  std::mt19937 gen(seed);
  std::vector<int> factorPool{1, 2, 3, 5, 7, 11, 13, 17, 19, 23};
  if (nDistinctFactors > factorPool.size()) {
    throw poprithms::test::error(
        "invalid nDistinctFactors, it should be less than " +
        std::to_string(factorPool.size()));
  }

  // generate the factors
  std::vector<int> factors;
  factors.reserve(nFactors);
  for (uint64_t i = 0; i < nFactors; ++i) {
    factors.push_back(
        factorPool[static_cast<uint64_t>(gen() % nDistinctFactors)]);
  }

  // Generate 2 shapes whose sizes are a product of factors
  std::vector<int64_t> shape0(l0, 1);
  std::vector<int64_t> shape1(l1, 1);
  for (uint64_t i = 0; i < nFactors; ++i) {
    shape0[gen() % l0] *= factors[i];
    shape1[gen() % l1] *= factors[i];
  }

  return {shape0, shape1};
}

Region
getRandomRegion(const Shape &sh, uint32_t seed, uint64_t maxSettDepth) {
  std::mt19937 gen(seed);
  std::vector<Sett> setts;
  setts.reserve(sh.rank_u64());
  for (uint64_t d = 0; d < sh.rank_u64(); ++d) {
    const auto depth = static_cast<uint64_t>(gen() % (maxSettDepth + 1));
    std::vector<Stripe> stripes;
    stripes.reserve(depth);
    auto maxPeriod = sh.dim(d);
    for (uint64_t s = 0; s < depth; ++s) {
      const auto period = std::max<int64_t>(
          1LL, static_cast<int64_t>(gen() % (maxPeriod + 1)));

      const auto on_unsigned = 1 + gen() % (period);
      const auto on          = static_cast<int64_t>(on_unsigned);

      const auto off = period - on;

      const auto phase_unsigned = gen() % (10 * period + 1);
      const auto phase          = static_cast<int64_t>(phase_unsigned);

      stripes.push_back({on, off, phase});
      maxPeriod = on;
    }
    setts.push_back(stripes);
  }
  return Region(sh, setts);
}

} // namespace nest
} // namespace memory
} // namespace poprithms
