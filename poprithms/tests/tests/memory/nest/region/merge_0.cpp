// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

int main() {
  using namespace poprithms::memory::nest;

  // Mergable along inner dimension:
  // 1,4,2 and 1,4,3 merge to give 2,3,2.

  Region r1({5, 6, 7},
            {{{{1, 4, 2}}}, {{{1, 1, 1}}}, Sett::createAlwaysOn()});
  Region r2({5, 6, 7},
            {{{{1, 4, 3}}}, {{{1, 1, 1}}}, Sett::createAlwaysOn()});
  auto merged = r1.merge(r2);
  if (!merged.full()) {
    throw poprithms::test::error("failed test, Regions should be merged");
  }
  if (!merged.first().equivalent(
          Region({5, 6, 7},
                 {{{{2, 3, 2}}}, {{{1, 1, 1}}}, Sett::createAlwaysOn()}))) {
    throw poprithms::test::error("failed test, merged Region is incorrect");
  }
}
