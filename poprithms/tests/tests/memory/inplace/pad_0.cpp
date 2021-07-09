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
  const auto v0              = Tensor::variable(graph, {3});
  const auto foo             = v0.pad({{{1}, {1}}}, false);
  const auto aliasGateNotPll = foo.closedAliasGate();
  aliasGateNotPll.modify();
  const auto aliasGatePll = v0.pad({{{1}, {1}}}, true).closedAliasGate();
  aliasGatePll.modify();

  const auto tryNotPll = graph.tryOpening({aliasGateNotPll, 0},
                                          CheckParallelWriteable::Yes,
                                          AllowMultiGateAlias::No);
  if (tryNotPll != OpeningStatus::NotParallelWriteable) {
    std::ostringstream oss;
    oss << "If this AliasGate is opened, the Tensor which is padded with "
        << " a broadcast constant would be modified."
        << "But CheckParallelWriteable::Yes is set, so this is not allowed. ";
    throw error(oss.str());
  }
  const auto tryPll = graph.tryOpening({aliasGatePll, 0},
                                       CheckParallelWriteable::Yes,
                                       AllowMultiGateAlias::No);
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
