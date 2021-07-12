// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/chain.hpp>

namespace {

using namespace poprithms::memory::chain;

void testRemoveIdentity0() {
  auto chain = Chain({10, 20});
  chain.reshape({10, 20});
  chain.expand({10, 20});
  chain.reduce({10, 20});
  chain.dimShuffle({{0, 1}});
  chain.reverse({});
  chain.slice({0, 0}, {10, 20});
  chain.canonicalized().confirmEqual(Chain({10, 20}));
  chain.canonicalized().confirmNotEqual(Chain({20, 20}));

  Chain diff({20, 10});
  diff.reshape({10, 20});
  chain.canonicalized().confirmNotEqual(diff);
}

} // namespace

int main() {
  testRemoveIdentity0();
  return 0;
}
