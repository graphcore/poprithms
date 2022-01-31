// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

using namespace poprithms::ndarray;
using namespace poprithms::memory::alias;

void testSubscript(const Shape &shape,
                   uint64_t index,
                   const Shape &expected) {
  Graph g;
  auto tensor = g.tensor(g.allocate(shape)).subscript(index);
  if (tensor.shape() != expected) {
    throw poprithms::test::error(
        "Failed subscript test: new shape is inconsistent with expected.");
  }
}

void testSubscriptError(const Shape &shape, uint64_t index) {
  Graph g;
  bool caught(false);
  try {
    auto tensor = g.tensor(g.allocate(shape)).subscript(index);
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Test succeeded unexpectedly with bad subscript args.");
  }
}

void testSubscript0() {
  testSubscript({2, 2, 2}, 1, {2, 2});
  testSubscript({1, 2, 3}, 0, {2, 3});
  testSubscript({5, 3, 1}, 3, {3, 1});
  testSubscript({5}, 2, {});
  testSubscript({1, 2, 3, 4}, 0, {2, 3, 4});
}

void testSubscript1() {
  testSubscriptError({2, 2}, 3);
  testSubscriptError({}, 0);
}

int main() {
  testSubscript0();
  testSubscript1();
}
