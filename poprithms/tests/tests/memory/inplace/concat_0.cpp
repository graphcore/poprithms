// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;
void test0() {

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
      cats2.push_back(Tensor::concat({vars.cend() - 2, vars.cend()}, 0)
                          .closedAliasGate());
    }
    if (i % 4 == 3) {
      cats4.push_back(Tensor::concat({cats2.cend() - 2, cats2.cend()}, 0)
                          .closedAliasGate());
    }
    if (i % 8 == 7) {
      cats8.push_back(Tensor::concat({cats4.cend() - 2, cats4.cend()}, 0)
                          .closedAliasGate());
    }
  }

  auto allCats = cats2;
  allCats.insert(allCats.end(), cats4.cbegin(), cats4.cend());
  allCats.insert(allCats.end(), cats8.cbegin(), cats8.cend());

  g.tryOpenings0(Tensor::opIds(allCats),
                 CheckParallelWriteable::Yes,
                 AllowMultiGateAlias::No);

  for (auto id : allCats) {
    if (id.aliasGateIsClosed()) {
      throw error("expected all concats to be inplaced");
    }
  }
}

void test1() {

  //           X0          .
  //        /     \        .
  //    modify  transpose  .
  //       \       |       .
  //        \     aliasGate      .
  //         \   /         .
  //         concat        .

  Graph g;
  auto X0 = Tensor::variable(g, {4, 4});
  auto u  = X0.modify();
  auto t  = X0.dimShuffle({{1, 0}});
  auto m  = t.closedAliasGate();
  g.constraint(m.opId(), u.opId());
  Tensor::concat({m, u}, 0);

  auto trial = g.tryOpenings0(
      {m.opId()}, CheckParallelWriteable::No, AllowMultiGateAlias::No);
  if (trial.size() != 1 || trial[0] != OpeningStatus::Cycle) {
    throw error("Opening the aliasGate makes the concat invalid. ");
  }
}

} // namespace

int main() {
  test0();
  test1();
}
