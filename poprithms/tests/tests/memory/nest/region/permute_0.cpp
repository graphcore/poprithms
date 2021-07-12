// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/permutation.hpp>

int main() {
  using namespace poprithms::memory::nest;

  Region r0({4, 5, 26},
            {{{{2, 1, 0}}}, {{{2, 2, 1}}}, {{{6, 3, 2}, {1, 1, 2}}}});

  const Permutation perm({1, 2, 0});

  const auto dimShuffled = r0.dimShuffle(perm);
  if (dimShuffled.shape() != Shape{5, 26, 4}) {
    throw poprithms::test::error(
        "Permuted Region has incorrect Shape, test failed");
  }

  for (uint64_t i = 0; i < 3; ++i) {
    if (!r0.sett(perm.get(i)).equivalent(dimShuffled.sett(i))) {
      throw poprithms::test::error("Permuted Sett failed test");
    }
  }
  return 0;
}
