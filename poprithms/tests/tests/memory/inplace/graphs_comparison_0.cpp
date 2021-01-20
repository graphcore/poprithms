// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void graphCopyTest0() {

  Graph g0;
  auto g1 = g0;
  if (g1 != g0) {
    throw error("Graphs should be equal after copy constructor invoked on "
                "uncontained Graph");
  }
  Shape s0{3, 4};
  g0.variable(s0);
  if (g1 == g0) {
    throw error("Graphs should NOT be equal after g0 has been modified");
  }

  auto g2 = g1;
  g1.variable(s0);
  if (g1 != g0) {
    throw error("Graphs should be equal again now, as g1 has also had the "
                "variable of shape {3,4} inserted");
  }

  g2.variable({4, 3});
  if (g1 != g0) {
    throw error("g2 has had a differently shaped variable inserted, and so "
                "should not compare equal");
  }

  auto g3 = g2;
  g2.setName("g2");
  g3.setName("g3");
  if (g2 == g3) {
    throw error("Graphs with different names should not compare equal");
  }
}

void graphMuxTest0() {
  Graph g0;
  const auto m0 = Tensor::variable(g0, {15}).closedMux();

  Graph g1;
  Tensor::variable(g1, {15}).openMux();

  if (g0 == g1) {
    throw error("The 2 Mux's are different : 1 is open and 1 is closed. The "
                "Graph comparison should have been different");
  }

  g0.tryOpening(Proposal(m0, 0), CheckParallelWriteable::Yes);
  if (g0 != g1) {
    throw error("g0 should have succeeded inplacing its Mux, thus removinf "
                "the only difference between the graphs g0 and g1. ");
  }
}

void graphVarTest0() {
  Graph g0;
  g0.variable({1, 2, 3});

  Graph g1;
  g1.constant({1, 2, 3});

  if (g0 == g1) {
    throw error(
        "g0 has a Variable of Shape {1,2,3}. whereas g1 has a Constant of "
        "that Shape. The Graphs should not compare equal");
  }
}

} // namespace

int main() {

  graphCopyTest0();
  graphMuxTest0();
  graphVarTest0();

  return 0;
}
