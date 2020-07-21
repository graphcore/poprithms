// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main() {

  using namespace poprithms::schedule::anneal;

  Graph g;
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  auto op2 = g.insertOp("op2");

  auto alloc = g.insertAlloc(100);
  g.insertOpAlloc(op0, alloc);
  g.insertOpAlloc(op2, alloc);

  g.insertConstraint(op0, op1);
  g.insertConstraint(op0, op2);

  // Due to alloc liveness, preferred schedule would be:
  //    op0,op2,op1
  // However, the following bin constraints should force the schedule to be:
  //    op0,op1,op2

  std::vector<std::vector<OpAddress>> bins(3);
  bins[0].push_back(op0);
  bins[0].push_back(op1);
  bins[2].push_back(op2);

  g.insertBinConstraints(bins, "test");

  g.initialize();
  g.minSumLivenessAnneal();
  auto op1_before_op2 = g.opToSchedule(op1) < g.opToSchedule(op2);
  if (!op1_before_op2) {
    throw error("Bin constraints should force op1  to be before op2");
  }
  return 0;
}