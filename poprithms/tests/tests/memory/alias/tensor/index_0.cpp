// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

using namespace poprithms::ndarray;
using namespace poprithms::memory::alias;

void testIndex(const std::vector<int64_t> &shape,
               const std::vector<uint64_t> &indices,
               const Shape &expected) {
  Graph g;
  auto tensor = g.tensor(g.allocate(Shape(shape))).index(indices);
  if (tensor.shape() != expected) {
    throw poprithms::test::error(
        "Failed index test: new shape is inconsistent with expected.");
  }
}

void testIndexError(const std::vector<int64_t> &shape,
                    const std::vector<uint64_t> &indices) {
  Graph g;
  bool caught(false);
  try {
    g.tensor(g.allocate(Shape(shape))).index(indices);
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Test succeeded unexpectedly with bad index args.");
  }
}

void compareIntersect(const Tensor &t1,
                      const Tensor &t2,
                      bool expectIntersect) {
  if (t1.intersectsWith(t2) != expectIntersect) {
    throw poprithms::test::error(
        "Intersection between t1 and t2 not as expected.");
  }
}

void testIndex0() {
  testIndex({2, 2, 2, 2}, {1, 1}, {2, 2});
  testIndex({1, 2, 3, 4}, {0, 1}, {3, 4});
  testIndex({1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});
  testIndex({1}, {0}, {});
  testIndex({1, 2, 3}, {0, 0, 0}, {});
  testIndex({}, {}, {});
}

void testIndex1() {
  testIndexError({1, 2}, {0, 0, 0});
  testIndexError({1, 1}, {3});
  testIndexError({}, {0});
}

void testIndex2() {
  Graph g;
  auto tensor = g.tensor(g.allocate({2, 3, 4}));
  testIndex({1, 2, 3, 4}, {0}, tensor.shape());
  testIndex({1, 2, 3, 4}, {0, 0}, tensor.subscript(0).shape());
  testIndex({1, 2, 3, 4}, {0, 0}, tensor.subscript(0).shape());
}

void testIndex3() {
  Graph g;
  auto tensor       = g.tensor(g.allocate({2, 3, 4, 5}));
  auto index0       = tensor.index({0});
  auto index1       = tensor.index({1});
  auto subscript0   = tensor.subscript(0);
  auto subscript1   = tensor.subscript(1);
  auto index0_0     = tensor.index({0, 0});
  auto subscript0_0 = subscript0.subscript(0);

  compareIntersect(index0, subscript0, true);
  compareIntersect(index1, subscript1, true);
  compareIntersect(index0, subscript1, false);
  compareIntersect(index0, index1, false);
  compareIntersect(index0_0, subscript0_0, true);
  compareIntersect(index0_0, subscript0, true);
  compareIntersect(index0, subscript0_0, true);
}

int main() {
  testIndex0();
  testIndex1();
  testIndex2();
  testIndex3();
}
