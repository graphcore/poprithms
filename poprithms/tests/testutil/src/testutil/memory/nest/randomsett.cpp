// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace poprithms {
namespace memory {
namespace nest {
Sett getRandom(bool shorten,
               int64_t depth,
               bool canonicalize,
               int seed,
               int max0) {

  std::mt19937 gen(seed);

  std::vector<Stripe> stripes;
  for (uint64_t i = 0; i < depth; ++i) {
    int64_t on{0};
    int64_t off{0};
    if (stripes.empty() || !shorten) {
      on  = gen() % max0;
      off = gen() % max0;
    } else {
      on  = gen() % std::max<int>(1, stripes.back().on());
      off = gen() % std::max<int>(1, stripes.back().on());
    }
    if (on + off == 0) {
      off = 1;
    }
    const auto phase = gen() % (on + off);
    stripes.push_back({on, off, static_cast<int64_t>(phase)});
  }

  return Sett(stripes, canonicalize);
}
} // namespace nest
} // namespace memory
} // namespace poprithms
