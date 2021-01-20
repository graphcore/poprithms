// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {
using namespace poprithms::memory::inplace;
void testDimShuffle0() {
  Graph g;

  //    x0  -> mux -> dimShuffle -> slice -> mux +
  //    |                                        + - concat -> mux -> unary.
  //  slice -> mux ------------------------------+
  //
  //  The 2 slices slice the exact same elements from x0.
  //

  const auto x0    = Tensor::variable(g, {2, 3, 5});
  const auto x0Mux = x0.closedMux();
  const auto d0    = x0Mux.dimShuffle({{1, 2, 0}});
  if (d0.shape() != Shape{3, 5, 2}) {
    throw error("dimShuffle shape incorrect");
  }
  const auto s0Mux  = d0.slice({2, 2, 1}, {3, 3, 2}).closedMux();
  const auto s1Mux  = x0.slice({1, 2, 2}, {2, 3, 3}).closedMux();
  const auto catMux = Tensor::concat({s0Mux, s1Mux}, 0).closedMux();
  catMux.modify();
  Tensors order{s1Mux, s0Mux, x0Mux, catMux};

  std::cout << g.tryOpenings0(Tensor::opIds(order),
                              CheckParallelWriteable::Yes)
            << std::endl;
  for (auto id : order) {
    if (id != catMux) {

      if (id.muxIsClosed()) {
        throw error("Expected all except cat to be inplace");
      }
    } else {
      if (id.muxIsOpen()) {
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
