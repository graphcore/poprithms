// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void test0() {
  Graph g;
  const auto c0 = Tensor::constant(g, {3, 3});
  const auto v0 = c0.reshape({9});
  const auto x0 = v0.closedAliasGate();
  x0.modify();

  {
    auto g0 = g;
    g0.tryOpening(
        {x0.opId(), 0}, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
    if (g0.aliasGateIsOpen(x0.opId())) {
      throw poprithms::test::error("inplace modified a constant");
    }
  }

  {
    auto g1 = g;
    g1.tryOpening(
        {x0.opId(), 0}, CheckParallelWriteable::No, AllowMultiGateAlias::No);
    if (g1.aliasGateIsClosed(x0.opId())) {
      throw poprithms::test::error("failed to inplace when obey=false");
    }
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
