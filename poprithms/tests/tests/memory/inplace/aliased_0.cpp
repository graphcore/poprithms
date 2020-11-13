// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;

void sliceTest0() {

  for (auto squareSize : {4, 5, 6}) {
    Graph g;
    const auto v0 = g.variable({10, 10});
    const auto s0 = g.slice_(v0, {0, 0}, {squareSize, squareSize});
    const auto s1 =
        g.slice_(v0, {10 - squareSize, 10 - squareSize}, {10, 10});
    const auto c0 = g.concat_({s0, s1}, 0);
    const auto x0 = g.unary(c0, AliasType::outplace());
    g.tryInplace({x0, AliasType::allInplace()}, CheckParallelWriteable::Yes);
    if (squareSize > 5) {
      if (g.aliasType(x0) != AliasType::outplace()) {
        throw error(
            "squares overlap, causing aliasing - should not be modifiable");
      }
    }

    else {
      if (g.aliasType(x0) == AliasType::outplace()) {
        throw error("squares do not overlap, it is modifiable");
      }
    }
  }
}

void expandTest0() {

  for (auto tryExpandFirst : {true, false}) {
    Graph g;
    const auto v0       = g.variable({1, 3, 1, 4});
    const auto expandId = g.expand(v0, AliasType::outplace(), {2, 3, 5, 4});
    const auto x1       = g.unary(expandId, AliasType::outplace());
    const auto order =
        tryExpandFirst ? TensorIds{expandId, x1} : TensorIds{x1, expandId};
    g.tryInplaces({{order[0], AliasType::allInplace()},
                   {order[1], AliasType::allInplace()}},
                  CheckParallelWriteable::Yes);
    if (g.aliasType(order[0]) != AliasType::allInplace() ||
        g.aliasType(order[1]) != AliasType::outplace()) {
      throw error(
          "In expand test, expected first attempted Op to be inplaced only");
    }
  }
}

} // namespace

int main() {
  sliceTest0();
  expandTest0();
  return 0;
}
