#include <iostream>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main() {
  using namespace poprithms::schedule::anneal;

  //
  //    X
  //   /
  //  X
  //
  Graph g;
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  g.insertConstraint(op0, op1);
  if (g.getTightPairs().size() != 1) {
    throw error("Expected 1 tight edge in this bipole graph");
  }

  //
  //    X
  //   / \
  //  X   X
  //   \ /
  //    X
  //
  auto op2 = g.insertOp("op2");
  g.insertConstraint(op0, op2);

  auto op3 = g.insertOp("op3");
  g.insertConstraint(op1, op3);
  g.insertConstraint(op2, op3);
  if (g.getTightPairs().size() != 0) {
    throw error("Expected 0 tight edge in this diamond");
  }

  //
  //    X
  //   / \
  //  X   X
  //   \ /
  //    X
  //    |
  //    X
  //    |
  //    X
  //
  auto op4 = g.insertOp("op4");
  auto op5 = g.insertOp("op5");
  g.insertConstraint(op3, op4);
  g.insertConstraint(op4, op5);
  if (g.getTightPairs().size() != 2) {
    throw error("Expected 2 tight edge in this tadpole");
  }

  return 0;
}