// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <random>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

std::vector<std::vector<uint64_t>>
getRandomEdges(uint64_t N, uint64_t E, uint64_t D, int seed) {

  std::vector<std::vector<uint64_t>> fwd(N);
  std::mt19937 gen(seed);
  std::vector<uint64_t> indices(N);
  std::iota(indices.begin(), indices.end(), 0);

  if (E > D) {
    throw poprithms::test::error(
        "E cannot be larger than D in edgemap::getRandomEdges");
  }
  if (D > N - 10) {
    throw poprithms::test::error(
        "D cannot be larger than N - 10 in edgemap::getRandomEdges");
  }
  auto nRando = N - D - 1;
  for (uint64_t i = 0; i < nRando; ++i) {
    fwd[i].reserve(E);
    std::sample(indices.begin() + i + 1,
                indices.begin() + i + 1 + D,
                std::back_inserter(fwd[i]),
                E,
                gen);
  }
  for (uint64_t i = nRando; i < N - 1; ++i) {
    fwd[i].push_back(i + 1);
  }

  return fwd;
}
} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
