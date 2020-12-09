// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <sstream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void testPad0() {
  Graph graph;
  const auto v0 = Tensor::variable(graph, {3});

  const auto muxNotPll = v0.pad({{{1}, {1}}}, false).closedMux();
  muxNotPll.unary();

  const auto muxPll = v0.pad({{{1}, {1}}}, true).closedMux();
  muxPll.unary();

  std::cout << graph << std::endl;

  const auto tryNotPll =
      graph.tryOpening({muxNotPll, 0}, CheckParallelWriteable::Yes);
  if (tryNotPll != OpeningStatus::NotParallelWriteable) {
    std::ostringstream oss;
    oss << "If this Mux is opened, the Tensor which is padded with "
        << " a broadcast constant would be modified."
        << "But CheckParallelWriteable::Yes is set, so this is not allowed. ";
    throw error(oss.str());
  }
  const auto tryPll =
      graph.tryOpening({muxPll, 0}, CheckParallelWriteable::Yes);
  if (tryPll != OpeningStatus::Valid) {
    throw error(
        "The Tensor padded with non-broadcast variable can be modified. ");
  }
}

} // namespace

int main() {
  testPad0();
  return 0;
}
