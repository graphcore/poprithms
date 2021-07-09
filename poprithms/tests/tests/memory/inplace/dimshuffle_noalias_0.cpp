// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {
using namespace poprithms::memory::inplace;
void testDimShuffle0() {
  Graph g;

  //    x0  -> aliasGate -> dimShuffle -> slice -> aliasGate +
  //    |                                        + - concat -> aliasGate ->
  //    unary.
  //  slice -> aliasGate ------------------------------+
  //
  //  The 2 slices slice the exact same elements from x0.
  //

  const auto x0          = Tensor::variable(g, {2, 3, 5});
  const auto x0AliasGate = x0.closedAliasGate();
  const auto d0          = x0AliasGate.dimShuffle({{1, 2, 0}});
  if (d0.shape() != Shape{3, 5, 2}) {
    throw error("dimShuffle shape incorrect");
  }
  const auto s0AliasGate = d0.slice({2, 2, 1}, {3, 3, 2}).closedAliasGate();
  const auto s1AliasGate = x0.slice({1, 2, 2}, {2, 3, 3}).closedAliasGate();
  const auto catAliasGate =
      Tensor::concat({s0AliasGate, s1AliasGate}, 0).closedAliasGate();
  catAliasGate.modify();
  Tensors order{s1AliasGate, s0AliasGate, x0AliasGate, catAliasGate};

  std::cout << g.tryOpenings0(Tensor::opIds(order),
                              CheckParallelWriteable::Yes,
                              AllowMultiGateAlias::No)
            << std::endl;
  for (auto id : order) {
    if (id != catAliasGate) {

      if (id.aliasGateIsClosed()) {
        throw error("Expected all except cat to be inplace");
      }
    } else {
      if (id.aliasGateIsOpen()) {
        throw error("Expected cat to be outplace (otherwise alias modified)");
      }
    }
  }
}

void testNoAlias0() {

  Graph g;
  const auto v0  = Tensor::variable(g, {5, 3});
  const auto v1  = Tensor::variable(g, {7, 11});
  const auto nax = Tensor::multi(g, {v0, v1}, {{1, 2}, {3, 4}, {5, 6}}, {});

  if (nax[1].shape() != Shape{3, 4}) {
    throw error("incorrect output Shape of NoAlias Op");
  }
}

} // namespace

int main() {
  testDimShuffle0();
  testNoAlias0();
  return 0;
}
