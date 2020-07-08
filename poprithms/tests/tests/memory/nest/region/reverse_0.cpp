// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>

int main() {

  using namespace poprithms::memory::nest;
  const Region r0({10, 11, 12},
                  {{{{6, 4, 1}}}, {{{8, 3, 0}, {1, 1, 0}}}, {{{1, 1, 0}}}});

  const auto flipped = r0.reverse({0, 1});
  if (flipped.shape() != r0.shape()) {
    throw error("Reverse should never change the Shape");
  }

  if (!flipped.sett(0).equivalent({{{6, 4, 3}}})) {
    throw error("Error reversing on dim 0 in test");
  }

  if (!flipped.sett(1).equivalent({{{8, 3, 3}, {1, 1, 1}}})) {
    throw error("Error reversing on dim 1 in test");
  }

  if (!flipped.sett(2).equivalent(r0.sett(2))) {
    throw error("Error (not) reversing on dim 2 in test");
  }
  return 0;
}
