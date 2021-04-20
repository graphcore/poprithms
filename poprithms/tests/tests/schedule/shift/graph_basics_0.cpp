// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <string>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>

// A test to assert that an error is thrown when an invalid OpAddress is used
// in a constraint
int test0() {
  using namespace poprithms::schedule::shift;
  Graph g;
  auto op0 = g.insertOp("Op0");
  auto op1 = g.insertOp("Op1");
  try {
    // OpId of "after" is too large
    g.insertConstraint(op0, op1 + 1);
  } catch (const poprithms::error::error &e) {
    return 1;
  }
  return 0;
}

int main() {
  using namespace poprithms::schedule::shift;
  if (test0() != 1) {
    throw error("Inserting constraint with non-existant Op was not caught");
  }
  return 0;
}
