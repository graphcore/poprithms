// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace {

using namespace poprithms::memory::chain;

void testReverseChain0() {

  const auto chain = Chain({10, 20, 30})
                         .reshape({5, 10, 120})
                         .dimShuffle({{1, 2, 0}})
                         .reverse(Dimensions({2}))
                         .settSample({{{{1, 1, 0}}}, {{{1, 1, 0}}}, {{{}}}});

  const auto revChain = chain.mirror();

  const auto expected =
      Chain({5, 60, 5})
          .settFillInto(
              Region({10, 120, 5}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}, {{{}}}}))
          .reverse(Dimensions({2}))
          .dimShuffle({{2, 0, 1}})
          .reshape({10, 20, 30});

  revChain.confirmEqual(expected);
}

} // namespace

int main() {
  testReverseChain0();
  return 0;
}
