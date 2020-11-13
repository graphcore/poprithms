// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;

void test0() {
  Graph g;
  const auto c0 = g.constant({3, 3});
  const auto v0 = g.reshape(c0, AliasType::allInplace(), {9});
  const auto x0 = g.unary(v0, AliasType::outplace());

  {
    auto g0 = g;
    g0.tryInplace({x0, AliasType::allInplace()}, CheckParallelWriteable::Yes);
    if (g0.aliasType(x0) != AliasType::outplace()) {
      throw error("inplace modified a constant");
    }
  }

  {
    auto g1 = g;
    g1.tryInplace({x0, AliasType::allInplace()}, CheckParallelWriteable::No);
    if (g1.aliasType(x0) == AliasType::outplace()) {
      throw error("failed to inplace when obey=false");
    }
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
