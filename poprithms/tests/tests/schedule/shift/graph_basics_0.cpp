// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <string>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>

namespace {
using namespace poprithms::schedule::shift;

// A test to assert that an error is thrown when an invalid OpAddress is used
// in a constraint
void test0() {
  Graph g;
  auto op0 = g.insertOp("Op0");
  auto op1 = g.insertOp("Op1");

  bool caught{false};
  try {
    // OpId of "after" is too large
    g.insertConstraint(op0, op1 + 1);
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Inserting constraint with non-existant Op was not caught");
  }
}

void test1() {

  //
  //     0 -> 1 -> 2 --+
  //          |        |
  //          v        v
  //          +------- 3

  {
    std::vector<std::vector<int>> fwd{{1}, {2, 3}, {3}, {}};
    Graph g(fwd);
  }

  {
    std::vector<std::vector<uint64_t>> fwd{{1}, {2, 3}, {3}, {}};
    Graph g(fwd);
  }

  bool caught{false};
  try {
    std::vector<std::vector<uint64_t>> fwd{{1}, {2, 3}, {3}, {1000}};
    Graph g(fwd);
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to detect invalid address in out edge");
  }
}
} // namespace

int main() {
  test0();
  test1();
  return 0;
}
