// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace {

using namespace poprithms::memory::chain;
using namespace poprithms::memory;

void testReverseChain0() {

  Chain chain({10, 20, 30});
  chain.reshape({5, 10, 120});
  chain.dimShuffle({{1, 2, 0}});
  chain.reverse(Dimensions({2}));
  chain.settSample(
      {{{{1, 1, 0}}}, {{{1, 1, 0}}}, nest::Sett::createAlwaysOn()});

  const auto revChain = chain.mirror();

  Chain expected({5, 60, 5});
  expected.settFillInto(
      Region({10, 120, 5},
             {{{{1, 1, 0}}}, {{{1, 1, 0}}}, nest::Sett::createAlwaysOn()}));
  expected.reverse(Dimensions({2}));
  expected.dimShuffle({{2, 0, 1}});
  expected.reshape({10, 20, 30});

  revChain.confirmEqual(expected);
}

} // namespace

int main() {
  testReverseChain0();
  return 0;
}
