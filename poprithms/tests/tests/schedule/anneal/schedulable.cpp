#include <iostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main() {

  using namespace poprithms::schedule::anneal;

  Graph g;
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  auto op2 = g.insertOp("op2");
  g.insertConstraint(op0, op1);
  g.insertConstraint(op1, op2);
  g.insertConstraint(op2, op0);
  g.finalize();
  if (g.isSchedulable()) {
    throw error("Triangle of dependencies is NOT schedulable");
  }

  g        = Graph();
  op0      = g.insertOp("op0");
  op1      = g.insertOp("op1");
  op2      = g.insertOp("op2");
  auto op3 = g.insertOp("op3");
  g.insertConstraint(op0, op1);
  g.insertConstraint(op0, op2);
  g.insertConstraint(op1, op3);
  g.insertConstraint(op2, op3);

  g.finalize();
  if (!g.isSchedulable()) {
    throw error("This diamond DAG IS schedulable");
  }

  return 0;
}
