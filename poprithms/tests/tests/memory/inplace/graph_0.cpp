// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

int main() {

  // Ensure that the Graph's constructors and copy operators are working

  using namespace poprithms::memory::inplace;

  Graph g;
  g.variable({3, 3});
  auto g2 = g;
  g2.constant({5, 5});
  Graph g3(g2);
  g3.variable({7, 7});
  g       = g3;
  auto g4 = std::move(g);
  g4.constant({11, 11});
  Graph g5(std::move(g4));

  if (g5.nTensors() != 4) {
    throw poprithms::test::error("expected g5 to have 4 Tensors");
  }

  return 0;
}
