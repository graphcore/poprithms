// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void graphCopyTest0() {

  Graph g0;
  auto g1 = g0;
  if (g1 != g0) {
    throw poprithms::test::error(
        "Graphs should be equal after copy constructor invoked on "
        "uncontained Graph");
  }
  Shape s0{3, 4};
  g0.variable(s0);
  if (g1 == g0) {
    throw poprithms::test::error(
        "Graphs should NOT be equal after g0 has been modified");
  }

  auto g2 = g1;
  g1.variable(s0);
  if (g1 != g0) {
    throw poprithms::test::error(
        "Graphs should be equal again now, as g1 has also had the "
        "variable of shape {3,4} inserted");
  }

  g2.variable({4, 3});
  if (g1 != g0) {
    throw poprithms::test::error(
        "g2 has had a differently shaped variable inserted, and so "
        "should not compare equal");
  }

  auto g3 = g2;
  g2.setName("g2");
  g3.setName("g3");
  if (g2 == g3) {
    throw poprithms::test::error(
        "Graphs with different names should not compare equal");
  }
}

void graphAliasGateTest0() {
  Graph g0;
  const auto m0 = Tensor::variable(g0, {15}).closedAliasGate();

  Graph g1;
  Tensor::variable(g1, {15}).openAliasGate();

  if (g0 == g1) {
    throw poprithms::test::error(
        "The 2 AliasGate's are different : 1 is open and 1 is closed. The "
        "Graph comparison should have been different");
  }

  g0.tryOpening(
      Proposal(m0, 0), CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
  if (g0 != g1) {
    throw poprithms::test::error(
        "g0 should have succeeded inplacing its AliasGate, thus removinf "
        "the only difference between the graphs g0 and g1. ");
  }
}

void graphVarTest0() {
  Graph g0;
  g0.variable({1, 2, 3});

  Graph g1;
  g1.constant({1, 2, 3});

  if (g0 == g1) {
    throw poprithms::test::error(
        "g0 has a Variable of Shape {1,2,3}. whereas g1 has a Constant of "
        "that Shape. The Graphs should not compare equal");
  }
}

} // namespace

int main() {

  graphCopyTest0();
  graphAliasGateTest0();
  graphVarTest0();

  return 0;
}
