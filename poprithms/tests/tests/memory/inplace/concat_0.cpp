// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

int main() {
  using namespace poprithms::memory::inplace;

  //
  //      x  x   x  x     x  x   x  x
  //      ====   ====     ====   ==== concat pairs of 2
  //      ===========     =========== ...
  //      =========================== concat all 8
  //

  Graph g;
  Tensors vars;
  Tensors cats2;
  Tensors cats4;
  Tensors cats8;
  for (uint64_t i = 0; i < 8; ++i) {
    vars.push_back(Tensor::variable(g, {1, 5}));
    if (i % 2 == 1) {
      cats2.push_back(
          Tensor::concat({vars.cend() - 2, vars.cend()}, 0).closedMux());
    }
    if (i % 4 == 3) {
      cats4.push_back(
          Tensor::concat({cats2.cend() - 2, cats2.cend()}, 0).closedMux());
    }
    if (i % 8 == 7) {
      cats8.push_back(
          Tensor::concat({cats4.cend() - 2, cats4.cend()}, 0).closedMux());
    }
  }

  auto allCats = cats2;
  allCats.insert(allCats.end(), cats4.cbegin(), cats4.cend());
  allCats.insert(allCats.end(), cats8.cbegin(), cats8.cend());

  g.tryOpenings0(Tensor::opIds(allCats), CheckParallelWriteable::Yes);

  for (auto id : allCats) {
    if (id.muxIsClosed()) {
      throw error("expected all concats to be inplaced");
    }
  }
  return 0;
}
