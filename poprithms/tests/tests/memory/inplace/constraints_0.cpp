// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {
using namespace poprithms::memory::inplace;
void test0() {

  //
  //   v0---------->  (unary_) -> x1
  //    |                ^
  //    |                |
  //    +--> (aliasGate) -> (unary_) -> x2
  //    |                ^
  //    |                |
  //    +--> (aliasGate) -> (unary_) -> x3
  //

  Graph g;
  const auto v0 = Tensor::variable(g, {3});

  const auto x1 = v0.modify();

  const auto x2m = v0.closedAliasGate();
  const auto x2  = x2m.modify();

  const auto x3m = v0.closedAliasGate();
  const auto x3  = x3m.modify();

  // confirm that inserting the same constraint multiple times is ok.
  for (int i = 0; i < 5; ++i) {
    g.constraint(v0.opId(), x3.opId(), x2.opId(), x1.opId());
  }

  g.tryOpening({x2m, 0}, CheckParallelWriteable::No);
  if (x2m.aliasGateIsOpen()) {
    throw error("cannot inplace x2, as constrained to be before x1");
  }
}

void testLateConstraint() {

  //          3
  //  v0 -> (aliasGate) -> (unary_) -> x0 -+
  //   |       ^                     |
  //   |       |                     |
  //   |       +-------+             +-- (cat_) -> (aliasGate) -> output
  //   |               |             |               1
  //   + -> (aliasGate) -> (unary_) -> x1 -+
  //          2
  //
  // aliasGate 1 ? yes
  // aliasGate 2 ? no
  // aliasGate 3 ? yes
  //

  Graph g;

  const auto v0           = Tensor::variable(g, {3});
  const auto x0aliasGate  = v0.closedAliasGate();
  const auto x1aliasGate  = v0.closedAliasGate();
  const auto x0_          = x0aliasGate.modify();
  const auto x1_          = x1aliasGate.modify();
  const auto cat          = Tensor::concat({x0_, x1_}, 0);
  const auto catAliasGate = cat.closedAliasGate();

  // inplace
  g.tryOpening({catAliasGate, 0}, CheckParallelWriteable::No);
  g.constraint(x1_.opId(), x0aliasGate.opId());

  // not inplace, as x0 must be before it
  g.tryOpening({x1aliasGate, 0}, CheckParallelWriteable::No);

  // inplace
  g.tryOpening({x0aliasGate, 0}, CheckParallelWriteable::No);
  if (x0aliasGate.aliasGateIsClosed() || x1aliasGate.aliasGateIsOpen()) {
    throw error("incorrect logic in testLateConstraint");
  }
}

} // namespace

int main() {
  test0();
  testLateConstraint();
}
