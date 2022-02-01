// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void testUpsample0() {
  Graph g;
  auto tensor = g.tensor(g.allocate({2}));

  // [xy] -> upsample(2,0) -> [xxyy]
  auto upsampled = tensor.upsample(2, 0);

  if (upsampled.shape() != Shape({4})) {
    throw poprithms::test::error(
        "Failed test: shape after upsampling not as expected.");
  }

  // [xxyy]
  //  ^^
  // upsampled[0] should intersect with upsampled[1]
  if (!upsampled.subscript(0).intersectsWith(upsampled.subscript(1))) {
    throw poprithms::test::error(
        "Failed test: upsampled[0] and upsampled[1] don't intersect.");
  }

  // [xxyy]
  //    ^^
  // upsampled[2] should intersect with upsampled[3]
  if (!upsampled.subscript(2).intersectsWith(upsampled.subscript(3))) {
    throw poprithms::test::error(
        "Failed test: upsampled[2] and upsampled[3] don't intersect.");
  }

  for (auto x = 0u; x < 2u; ++x) {
    for (auto y = 2u; y < 4u; ++y) {
      if (upsampled.subscript(x).intersectsWith(upsampled.subscript(y))) {
        std::ostringstream oss;
        oss << "Failed test: upsampled[" << x << "] and upsampled[" << y
            << "] are (erroneously) intersecting.";
        throw poprithms::test::error(oss.str());
      }
    }
  }
}

// Only comparing shape using this helper.
void testUpsampleShape(const Shape &shape,
                       uint64_t scale,
                       uint64_t dim,
                       const Shape &expected) {
  Graph g;
  auto tensor = g.tensor(g.allocate(shape));

  auto upsampled = tensor.upsample(scale, dim);

  if (upsampled.shape() != expected) {
    throw poprithms::test::error(
        "Failed test: shape after upsampling not as expected.");
  }
}

void testUpsampleError(const Shape &shape, uint64_t scale, uint64_t dim) {
  Graph g;
  auto tensor = g.tensor(g.allocate(shape));

  bool caught{false};

  try {
    auto upsampled = tensor.upsample(scale, dim);
    (void)upsampled;
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    std::ostringstream oss;
    oss << "Unexpected successful test: {";
    for (const auto &shapeDim : shape.get()) {
      oss << shapeDim;
      if (shapeDim != *std::prev(shape.get().end())) {
        oss << ", ";
      }
    }
    oss << "}.upsample(" << scale << ", " << dim << ") succeeded.";
    throw poprithms::test::error(oss.str());
  }
}

void testUpsample1() {
  testUpsampleShape({2, 2}, 3, 0, {6, 2});
  testUpsampleShape({2, 2}, 3, 1, {2, 6});
  testUpsampleShape({2, 2, 2}, 5, 2, {2, 2, 10});
  testUpsampleShape({2, 2}, 1, 0, {2, 2});
  testUpsampleShape({2, 2}, 1, 1, {2, 2});
  testUpsampleShape({2}, 0, 0, {0});
  testUpsampleShape({2, 2}, 0, 0, {0, 2});
  testUpsampleShape({2, 2}, 0, 1, {2, 0});
}

void testUpsample2() {
  testUpsampleError({2, 2}, 1, 4);
  testUpsampleError({}, 1, 0);
  testUpsampleError({}, 1, 4);
}

int main() {
  testUpsample0();
  testUpsample1();
  testUpsample2();
  return 0;
}
