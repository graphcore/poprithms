// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {
using namespace poprithms::memory::inplace;
void test0() {

  //
  //   v0---------->  (unary_) -> x1
  //    |                ^
  //    |                |
  //    +--> (mux) -> (unary_) -> x2
  //    |                ^
  //    |                |
  //    +--> (mux) -> (unary_) -> x3
  //

  Graph g;
  const auto v0 = Tensor::variable(g, {3});

  const auto x1 = v0.unary();

  const auto x2m = v0.closedMux();
  const auto x2  = x2m.unary();

  const auto x3m = v0.closedMux();
  const auto x3  = x3m.unary();

  // confirm that inserting the same constraint multiple times is ok.
  for (int i = 0; i < 5; ++i) {
    g.constraint(v0.opId(), x3.opId(), x2.opId(), x1.opId());
  }

  g.tryOpening({x2m, 0}, CheckParallelWriteable::No);
  if (x2m.muxIsOpen()) {
    throw error("cannot inplace x2, as constrained to be before x1");
  }
}

void testLateConstraint() {

  //          3
  //  v0 -> (mux) -> (unary_) -> x0 -+
  //   |       ^                     |
  //   |       |                     |
  //   |       +-------+             +-- (cat_) -> (mux) -> output
  //   |               |             |               1
  //   + -> (mux) -> (unary_) -> x1 -+
  //          2
  //
  // mux 1 ? yes
  // mux 2 ? no
  // mux 3 ? yes
  //

  Graph g;

  const auto v0     = Tensor::variable(g, {3});
  const auto x0mux  = v0.closedMux();
  const auto x1mux  = v0.closedMux();
  const auto x0_    = x0mux.unary();
  const auto x1_    = x1mux.unary();
  const auto cat    = Tensor::concat({x0_, x1_}, 0);
  const auto catMux = cat.closedMux();

  // inplace
  g.tryOpening({catMux, 0}, CheckParallelWriteable::No);
  g.constraint(x1_.opId(), x0mux.opId());

  // not inplace, as x0 must be before it
  g.tryOpening({x1mux, 0}, CheckParallelWriteable::No);

  // inplace
  g.tryOpening({x0mux, 0}, CheckParallelWriteable::No);
  if (x0mux.muxIsClosed() || x1mux.muxIsOpen()) {
    throw error("incorrect logic in testLateConstraint");
  }
}

} // namespace

int main() {
  test0();
  testLateConstraint();
}
