// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/permutation.hpp>

int main() {
  using namespace poprithms::memory::nest;

  Region r0({4, 5, 26},
            {{{{2, 1, 0}}}, {{{2, 2, 1}}}, {{{6, 3, 2}, {1, 1, 2}}}});

  const Permutation perm({1, 2, 0});

  const auto permuted = r0.permute(perm);
  if (permuted.shape() != Shape{5, 26, 4}) {
    throw error("Permuted Region has incorrect Shape, test failed");
  }

  for (uint64_t i = 0; i < 3; ++i) {
    if (!r0.sett(perm.get(i)).equivalent(permuted.sett(i))) {
      throw error("Permuted Sett failed test");
    }
  }
  return 0;
}
