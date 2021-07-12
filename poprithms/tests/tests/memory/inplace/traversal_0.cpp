// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <sstream>

#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void testTraversal0() {
  Graph graph;
  const auto v0 = Tensor::variable(graph, {3});
  graph.multi({v0.id(), v0.id(), v0.id()}, {{}, {}, {}, {}}, {});
  if (poprithms::common::multiout::depthFirstForward(
          graph, {v0.id()}, [](auto) { return true; })
          .size() != 12) {
    throw poprithms::test::error("3 inputs, 4 outputs: 12 paths.");
  }
}

} // namespace

int main() {
  testTraversal0();
  return 0;
}
