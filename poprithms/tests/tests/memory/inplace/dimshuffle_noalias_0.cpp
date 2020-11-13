// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {
using namespace poprithms::memory::inplace;
void testDimShuffle0() {
  Graph g;

  const auto x0 = g.variable({2, 3, 5});
  const auto d0 = g.dimShuffle(x0, AliasType::outplace(), {{1, 2, 0}});
  if (g.shape(d0) != Shape{3, 5, 2}) {
    throw error("dimShuffle shape incorrect");
  }
  const auto s0  = g.slice(d0, AliasType::outplace(), {2, 2, 1}, {3, 3, 2});
  const auto s1  = g.slice(x0, AliasType::outplace(), {1, 2, 2}, {2, 3, 3});
  const auto cat = g.concat({s0, s1}, AliasType::outplace(), 0);
  const auto u0  = g.unary(cat, AliasType::outplace());

  TensorIds order{s1, s0, d0, u0, cat};

  auto g0 = g;
  g0.tryInplaces(Graph::createProposalsAllInplace(order),
                 CheckParallelWriteable::Yes);
  for (TensorId id : order) {
    if (id != cat) {
      if (g0.aliasType(id) == AliasType::outplace()) {
        throw error("Expected all except cat to be inplace");
      }
    } else {
      if (g0.aliasType(id) != AliasType::outplace()) {
        throw error("Expected cat to be outplace (otherwise alias modified)");
      }
    }
  }
}

//   OpId  Name         OpType          InTensors  InOps  OutIndex
//   ----- ------------ --------------- ---------- ------ ---------
//   0                  Alloc(color=1)  ()         ()     0
//   1                  Alloc(color=1)  ()         ()     0
//   2     myComplexOp  NoAlias         (0,1)      (0,1)  0
//                                                        1
//                                                        2

void testNoAlias0() {

  Graph g;
  const auto v0  = g.variable({5, 3});
  const auto v1  = g.variable({7, 11});
  const auto nax = g.noAlias({v0, v1}, {{1, 2}, {3, 4}, {5, 6}});
  g.setName(nax, "myComplexOp");

  if (g.shape({nax, 1}) != Shape{3, 4}) {
    throw error("incorrect output Shape of NoAlias Op");
  }
}

} // namespace

int main() {
  testDimShuffle0();
  testNoAlias0();
  return 0;
}
