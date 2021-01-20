// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace {

using namespace poprithms::memory::chain;

void testRemoveIdentity0() {
  auto chain = Chain({10, 20})
                   .reshape({10, 20})
                   .expand({10, 20})
                   .reduce({10, 20})
                   .dimShuffle({{0, 1}})
                   .reverse({})
                   .slice({0, 0}, {10, 20});
  chain.removeIdentity().confirmEqual(Chain({10, 20}));
  chain.removeIdentity().confirmNotEqual(Chain({20, 20}));
  chain.removeIdentity().confirmNotEqual(Chain({20, 10}).reshape({10, 20}));
}

} // namespace

int main() {
  testRemoveIdentity0();
  return 0;
}
