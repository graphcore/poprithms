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
  //  slice -> concat -> aliasGate -> unary
  //
  //  aliasGate should only open if slices don't intersect.

  for (auto squareSize : {4, 5, 6}) {
    Graph g;
    const auto v0 = Tensor::variable(g, {10, 10});

    // Slice of lower-left corner
    const auto s0 = v0.slice({0, 0}, {squareSize, squareSize});

    // Slice of upper-right corner
    const auto s1 = v0.slice({10 - squareSize, 10 - squareSize}, {10, 10});

    // Concatenation of the 2 slices. They intersect if squareSize > 5.
    const auto x0 = Tensor::concat({s0, s1}, 0).closedAliasGate();

    x0.modify();
    g.tryOpening({x0.opId(), 0}, CheckParallelWriteable::Yes);

    if (squareSize > 5 && x0.aliasGateIsOpen()) {
      throw error(
          "Squares intersect, opening of aliasGate should not happen.");
    }

    else if (squareSize <= 5 && x0.aliasGateIsClosed()) {
      throw error(
          "Squares don't intersect, opening of aliasGate should happen.");
    }
  }
}

/**
 *  X -> expand -> aliasGate -> reshape -> aliasGate -> unary
 *
 *  With CheckParallelWriteable::Yes, one of the aliasGates must be closed,
 *  otherwise the modifier modifies a non-parallel writeable Tensor.
 * */
void expandTest0() {

  for (auto tryExpandFirst : {true, false}) {
    Graph g;
    const auto aliasGate0 = Tensor::variable(g, {1, 3, 1, 4})
                                .expand({2, 3, 5, 4})
                                .closedAliasGate();
    const auto aliasGate1 = aliasGate0.flatten().closedAliasGate();
    aliasGate1.modify();

    Tensors order{aliasGate0, aliasGate1};
    if (tryExpandFirst) {
      std::swap(order[0], order[1]);
    }
    g.tryOpenings0(Tensor::opIds(order), CheckParallelWriteable::Yes);

    const auto open0 = order[0].aliasGateIsOpen();
    const auto open1 = order[1].aliasGateIsOpen();

    if (!open0 || open1) {
      std::ostringstream oss;
      oss << "was the aliasGate just after expand tried first ? "
          << tryExpandFirst << ". Was the first aliasGate tried open ? "
          << open0 << ". Was the second aliasGate tried open ? " << open1
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
