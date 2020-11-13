// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <sstream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;

void testPad0() {

  Graph g;
  const auto a0 = g.variable({5, 5});
  // const auto lAndU = std::array<std::vector<int64_t>, 2>{{{1,1}, {1,1}}};

  const auto u0 = g.unary(a0, AliasType::outplace());

  //  const auto p0 = g.pad(u0, AliasType::outplace(), {{{1, 1}, {1, 1}}},
  //  true); const auto p1 = g.pad(a0, AliasType::outplace(), {{{1, 1}, {1,
  //  1}}}, true);

  const auto p0 = g.flatten(u0, AliasType::outplace());
  const auto p1 = g.flatten(a0, AliasType::outplace());

  const auto u1 = g.unary(p1, AliasType::outplace());
  g.binary(p0, u1, AliasType::outplace());

  std::cout << g << std::endl;
  std::cout << g.tryInplaces(
      Graph::createProposalsAllInplace({u0, u1, p0, p1}),
      CheckParallelWriteable::No);
  std::cout << g << std::endl;
}

} // namespace

int main() {
  testPad0();
  return 0;
}
