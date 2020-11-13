// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {
using namespace poprithms::memory::inplace;
void test0() {

  Graph g;
  const auto v0 = g.variable({3});

  const auto x1 = g.unary(v0, AliasType::allInplace());
  const auto x2 = g.unary(v0, AliasType::outplace());
  const auto x3 = g.unary(v0, AliasType::outplace());

  // confirm that inserting the same constraint multiple times is ok.
  for (int i = 0; i < 5; ++i) {
    g.constraint(v0, x3, x2, x1);
  }

  g.tryInplace({x2, AliasType::allInplace()}, CheckParallelWriteable::No);
  if (g.aliasType(x2) == AliasType::allInplace()) {
    throw error("cannot inplace x2, as constrained to be before x1");
  }
}

void testLateConstraint() {

  Graph g;

  const auto v0  = g.variable({3});
  const auto x0  = g.unary(v0, AliasType::outplace());
  const auto x1  = g.unary(v0, AliasType::outplace());
  const auto cat = g.concat({x0, x1}, AliasType::outplace(), 0);

  // inplace
  g.tryInplace({cat, AliasType::allInplace()}, CheckParallelWriteable::No);
  g.constraint(x1, x0);

  // not inplace, as x0 must be before it
  g.tryInplace({x1, AliasType::allInplace()}, CheckParallelWriteable::No);

  // inplace
  g.tryInplace({x0, AliasType::allInplace()}, CheckParallelWriteable::No);
  std::cout << g << std::endl;
  if (g.aliasType(x1) != AliasType::outplace() ||
      g.aliasType(x0) == AliasType::outplace()) {
    throw error("incorrect logic in testLateConstraint");
  }
}

} // namespace

int main() {
  test0();
  testLateConstraint();
}
