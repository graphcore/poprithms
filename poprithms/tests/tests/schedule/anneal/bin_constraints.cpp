// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

void skipping_bin_test() {
  using namespace poprithms::schedule::anneal;

  Graph g;
  const auto ops  = g.insertOps({"op0", "op1", "op2"});
  const auto allo = g.insertAlloc(100);
  g.insertOpAlloc({ops[0], ops[1]}, allo);
  g.insertConstraints({{ops[0], ops[1]}, {ops[0], ops[2]}});

  // Due to alloc liveness, preferred schedule would be:
  //    op0,op2,op1
  // However, the following bin constraints should force the schedule to be:
  //    op0,op1,op2

  g.insertBinConstraints({{ops[0], ops[1]}, {}, {ops[2]}}, "test");

  g.initialize();
  g.minSumLivenessAnneal();
  auto op1_before_op2 = g.opToSchedule(ops[1]) < g.opToSchedule(ops[2]);
  if (!op1_before_op2) {
    throw error("Skipping bin constraints should force op1 to be before op2");
  }
}

void multiple_bin_test() {
  using namespace poprithms::schedule::anneal;

  Graph g;
  const auto ops  = g.insertOps({"op0", "op1", "op2"});
  const auto allo = g.insertAlloc(100);
  g.insertOpAlloc({ops[0], ops[1]}, allo);
  g.insertConstraints({{ops[0], ops[1]}, {ops[0], ops[2]}});

  // Due to alloc liveness, preferred schedule would be:
  //    op0,op2,op1
  // However, the following nested bin constraints should force the schedule
  // to be:
  //    op0,op1,op2

  g.insertBinConstraints({{ops[0]}, {ops[1]}}, "phases");
  g.insertBinConstraints({{ops[0], ops[1]}, {ops[2]}}, "context");

  g.initialize();
  g.minSumLivenessAnneal();
  auto op1_before_op2 = g.opToSchedule(ops[1]) < g.opToSchedule(ops[2]);
  if (!op1_before_op2) {
    throw error("Multiple bin constraints should force op1 to be before op2");
  }
}

int main() {
  skipping_bin_test();

  multiple_bin_test();

  return 0;
}