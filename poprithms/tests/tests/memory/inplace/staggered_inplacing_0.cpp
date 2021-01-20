// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void testStaggered0() {

  //
  //                       [m0]
  //  x0  -- dimShuffle -- mux -- flatten
  //    \                            |
  //    reverse                     unary
  //       \
  //       mux [m1]
  //         \
  //         reshape
  //            \
  //            unary
  //              \
  //              unary [u0]
  //
  Graph g;
  const auto x0 = Tensor::variable(g, {3, 3});
  const auto m0 = x0.dimShuffle({{1, 0}}).closedMux();
  auto r0       = g.tryOpening({m0, 0}, CheckParallelWriteable::No);
  if (r0 != OpeningStatus::Valid) {
    throw error("No reason for r0 to have not been opened");
  }

  m0.flatten().modify();
  const auto m1 = x0.reverse(0).closedMux();
  const auto u1 = m1.reshape({3, 1, 3, 1}).modify().modify();

  const auto u1Alis = u1.allAliases();
  if (u1Alis.size() != 4) {
    throw error("Expected 4 aliases of u1 : output of mux, output of "
                "reshape, output of first unary, output of second unary. ");
  }

  const auto r1 = g.tryOpening({m1, 0}, CheckParallelWriteable::No);
  if (r1 != OpeningStatus::Cycle) {
    throw error("Opening the second mux creates 2 modifiers of x0. Thus "
                "expected the Status to be Cycle");
  }
}

} // namespace

int main() {
  testStaggered0();
  return 0;
}
