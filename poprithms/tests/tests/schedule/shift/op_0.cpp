// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <string>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>

namespace {

using namespace poprithms::schedule::shift;

void test1() {
  Graph g;
  auto foo = g.insertOp("Foo");
  auto bar = g.insertOp("Bar");
  g.insertConstraint(foo, bar);
  g.insertConstraint(foo, bar);
  if (g.getOp(foo).nOuts() != 1 || g.getOp(foo).getOuts().size() != 1) {
    throw error(
        "Duplicated constraints should be removed during construction");
  }
}

void test2() {

  Graph g;
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  auto op2 = g.insertOp("op2");
  auto op3 = g.insertOp("op3");
  auto op4 = g.insertOp("op4");

  g.insertConstraints({

      {op2, op0}, {op2, op4}, {op2, op3}, {op2, op1}}

  );

  auto outs   = g.getOp(op2).getOuts();
  auto sorted = outs;
  std::sort(sorted.begin(), sorted.end());
  if (outs.size() != 4) {
    throw error("Expected 4 outputs of op2");
  }

  if (sorted != outs) {
    throw error(
        "Constraints should be sorted at all times, done during insertion");
  }
}

void test0() {

  Graph g;

  uint64_t nOps = 5;
  for (uint64_t i = 0; i < nOps; ++i) {
    g.insertOp("Op" + std::to_string(i));
  }

  for (uint64_t i = 0; i < nOps - 1; ++i) {
    g.insertConstraint(i, nOps - 1);
  }

  auto g2 = g;
  for (uint64_t i = 0; i < nOps; ++i) {
    if (g2.getOp(i) != g.getOp(i)) {
      throw error("Expect Ops in copied Graph to compare to equal");
    }
  }

  for (uint64_t i = 0; i < nOps - 1; ++i) {
    if (!g.getOp(i).hasOut(nOps - 1) || !g.getOp(nOps - 1).hasIn(i)) {
      throw error("Unexpected in/out");
    }
  }

  Op op0(1000, "standaloneOp");
  op0.insertIn(1);
  op0.insertIn(3);
  op0.insertIn(2);
  op0.insertIn(4);
  if (!op0.hasIn(2)) {
    throw error("2 in an input to op0");
  }
  op0.removeIn(2);
  if (op0.hasIn(2)) {
    throw error("2 has been removed as an input to op0");
  }
}

} // namespace

int main() {

  test0();
  test1();
  test2();

  return 0;
}
