// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void sliceTest0() {

  //  X ->  slice
  //  |         |
  //  v         v
  //  slice -> concat -> mux -> unary
  //
  //  mux should only open if slices don't intersect.

  for (auto squareSize : {4, 5, 6}) {
    Graph g;
    const auto v0 = Tensor::variable(g, {10, 10});

    // Slice of lower-left corner
    const auto s0 = v0.slice({0, 0}, {squareSize, squareSize});

    // Slice of upper-right corner
    const auto s1 = v0.slice({10 - squareSize, 10 - squareSize}, {10, 10});

    // Concatenation of the 2 slices. They intersect if squareSize > 5.
    const auto x0 = Tensor::concat({s0, s1}, 0).closedMux();

    x0.modify();
    g.tryOpening({x0.opId(), 0}, CheckParallelWriteable::Yes);

    if (squareSize > 5 && x0.muxIsOpen()) {
      throw error("Squares intersect, opening of mux should not happen.");
    }

    else if (squareSize <= 5 && x0.muxIsClosed()) {
      throw error("Squares don't intersect, opening of mux should happen.");
    }
  }
}

/**
 *  X -> expand -> mux -> reshape -> mux -> unary
 *
 *  With CheckParallelWriteable::Yes, one of the muxs must be closed,
 *  otherwise the modifier modifies a non-parallel writeable Tensor.
 * */
void expandTest0() {

  for (auto tryExpandFirst : {true, false}) {
    Graph g;
    const auto mux0 =
        Tensor::variable(g, {1, 3, 1, 4}).expand({2, 3, 5, 4}).closedMux();
    const auto mux1 = mux0.flatten().closedMux();
    mux1.modify();

    Tensors order{mux0, mux1};
    if (tryExpandFirst) {
      std::swap(order[0], order[1]);
    }
    g.tryOpenings0(Tensor::opIds(order), CheckParallelWriteable::Yes);

    const auto open0 = order[0].muxIsOpen();
    const auto open1 = order[1].muxIsOpen();

    if (!open0 || open1) {
      std::ostringstream oss;
      oss << "was the mux just after expand tried first ? " << tryExpandFirst
          << ". Was the first mux tried open ? " << open0
          << ". Was the second mux tried open ? " << open1
          << ". In this test, "
          << "expected first attempted Op to be inplaced only. ";
      throw error(oss.str());
    }
  }
}

} // namespace

int main() {
  sliceTest0();
  expandTest0();
  return 0;
}
