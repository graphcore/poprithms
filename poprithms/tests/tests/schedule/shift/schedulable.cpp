// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

int main() {

  using namespace poprithms::schedule::shift;

  Graph g;
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  auto op2 = g.insertOp("op2");
  g.insertConstraint(op0, op1);
  g.insertConstraint(op1, op2);
  g.insertConstraint(op2, op0);
  if (ScheduledGraph::isSchedulable(g)) {
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
  if (!ScheduledGraph::isSchedulable(g)) {
    throw error("This diamond DAG IS schedulable");
  }

  return 0;
}
