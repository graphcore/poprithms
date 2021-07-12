// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

void skipping_bin_test() {
  using namespace poprithms::schedule::shift;

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

  ScheduledGraph sg(std::move(g));
  auto op1_before_op2 = sg.opToSchedule(ops[1]) < sg.opToSchedule(ops[2]);
  if (!op1_before_op2) {
    throw poprithms::test::error(
        "Skipping bin constraints should force op1 to be before op2");
  }
}

void multiple_bin_test() {
  using namespace poprithms::schedule::shift;

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

  ScheduledGraph sg(std::move(g));
  auto op1_before_op2 = sg.opToSchedule(ops[1]) < sg.opToSchedule(ops[2]);
  if (!op1_before_op2) {
    throw poprithms::test::error(
        "Multiple bin constraints should force op1 to be before op2");
  }
}

int main() {
  skipping_bin_test();

  multiple_bin_test();

  return 0;
}
