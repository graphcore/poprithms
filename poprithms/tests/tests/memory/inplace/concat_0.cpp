// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

int main() {
  using namespace poprithms::memory::inplace;

  //
  //      x  x   x  x     x  x   x  x
  //      ====   ====     ====   ==== concat pairs of 2
  //      ===========     =========== ...
  //      =========================== concat all 8
  //

  Graph g;
  TensorIds vars;
  TensorIds cats2;
  TensorIds cats4;
  TensorIds cats8;
  for (uint64_t i = 0; i < 8; ++i) {
    vars.push_back(g.variable({1, 5}));
    if (i % 2 == 1) {
      cats2.push_back(
          g.concat({vars.cend() - 2, vars.cend()}, AliasType::outplace(), 0));
    }
    if (i % 4 == 3) {
      cats4.push_back(g.concat(
          {cats2.cend() - 2, cats2.cend()}, AliasType::outplace(), 0));
    }
    if (i % 8 == 7) {
      cats8.push_back(g.concat(
          {cats4.cend() - 2, cats4.cend()}, AliasType::outplace(), 0));
    }
  }

  TensorIds allCats = cats2;
  allCats.insert(allCats.end(), cats4.cbegin(), cats4.cend());
  allCats.insert(allCats.end(), cats8.cbegin(), cats8.cend());

  g.tryInplaces(Graph::createProposalsAllInplace(allCats),
                CheckParallelWriteable::Yes);

  for (auto id : allCats) {
    if (g.aliasType(id) == AliasType::outplace()) {
      throw error("expected all concats to be inplaced");
    }
  }
  return 0;
}
