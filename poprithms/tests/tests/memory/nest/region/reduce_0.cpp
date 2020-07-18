// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>

int main() {
  using namespace poprithms::memory::nest;

  Region r0({2, 3, 4, 5},
            {{{{1, 1, 0}}}, {{{1, 1, 0}}}, {{{1, 1, 0}}}, {{{1, 1, 0}}}});

  const auto red0 = r0.reduce({1, 4, 1});
  if (!red0.equivalent({{1, 4, 1}, {{{{}}}, {{{1, 1, 0}}}, {{{}}}}})) {
    throw error("Reduction not as expected in test 0");
  }

  const auto red1 = r0.reduce({1, 3, 1, 5});
  if (!red1.equivalent(
          {{1, 3, 1, 5}, {{{{}}}, {{{1, 1, 0}}}, {{{}}}, {{{1, 1, 0}}}}})) {
    throw error("Reduction not as expected in test 1");
  }

  const auto red2 = r0.reduce({1});
  if (!red2.equivalent({{1}, {{{{}}}}})) {
    throw error("Reduction not as expected in test 1");
  }

  const auto red3 = r0.reduce({});
  if (!red3.equivalent({{}, {}})) {
    throw error("Reduction not as expected in test 1");
  }
}
