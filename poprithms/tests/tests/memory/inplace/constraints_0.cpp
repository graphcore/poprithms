// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
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

  g.tryOpening({x2m, 0}, CheckParallelWriteable::No, AllowMultiGateAlias::No);
  if (x2m.aliasGateIsOpen()) {
    throw poprithms::test::error(
        "cannot inplace x2, as constrained to be before x1");
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
  g.tryOpening(
      {catAliasGate, 0}, CheckParallelWriteable::No, AllowMultiGateAlias::No);
  g.constraint(x1_.opId(), x0aliasGate.opId());

  // not inplace, as x0 must be before it
  g.tryOpening(
      {x1aliasGate, 0}, CheckParallelWriteable::No, AllowMultiGateAlias::No);

  // inplace
  g.tryOpening(
      {x0aliasGate, 0}, CheckParallelWriteable::No, AllowMultiGateAlias::No);
  if (x0aliasGate.aliasGateIsClosed() || x1aliasGate.aliasGateIsOpen()) {
    throw poprithms::test::error("incorrect logic in testLateConstraint");
  }
}

void testConstraintsNotSetInPartialOpening() {
  // Construct a "diamond" test case with an extra constraint across the two
  // branches. This gives a test case where opening ag0 will not change the
  // schedule but will result in new constraints.
  Graph g;
  auto x   = Tensor::variable(g, {1}).closedAliasGate();
  auto ag0 = x.closedAliasGate();
  auto ag1 = x.closedAliasGate();
  auto c0  = ag0.modify();
  auto c1  = ag1.modify();
  g.concat(Tensor::tensorIds({c0, c1}), 0);
  g.constraint(c1.id(), ag0.id());

  // Save original graph.
  Graph h = g;

  Proposal p{ag0, 0};

  const auto status = g.tryOpeningPartial(
      p, CheckParallelWriteable::No, AllowMultiGateAlias::No);

  if (!status.isValid()) {
    throw poprithms::test::error(
        "incorrect logic in testConstraintsNotSetInPartialOpening: "
        "Opening was invalid.");
  }

  g.backoutOpening(p);

  if (h != g) {
    throw poprithms::test::error(
        "invariant g == g.tryOpeningPartial(p).backoutOpening(p) has "
        "failed.");
  }
}

} // namespace

int main() {
  test0();
  testLateConstraint();
  testConstraintsNotSetInPartialOpening();
}
