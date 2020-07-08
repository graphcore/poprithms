// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/util/error.hpp>
#include <poprithms/util/shape.hpp>

namespace {
using namespace poprithms::util;

void assertNumpyBroadcast(const std::vector<int64_t> &a,
                          const std::vector<int64_t> &b,
                          const std::vector<int64_t> &expected) {
  const auto out = Shape::numpyBinary(a, b);
  if (out != expected) {
    std::ostringstream oss;
    oss << "Failed in assertNumpyBroadcast";
    throw error(oss.str());
  }
}

void assertRowMajorIndex(const Shape &shape,
                         const std::vector<int64_t> &point,
                         int64_t expected) {
  if (shape.getRowMajorIndex(point) != expected) {
    throw error("Failed in assertRowMajorIndex");
  }
}

void testNumpyBinary0() {
  assertNumpyBroadcast({2, 3, 1}, {2, 3, 4}, {2, 3, 4});
  assertNumpyBroadcast({1, 3, 1}, {2, 1, 4}, {2, 3, 4});
  assertNumpyBroadcast({1, 3, 1}, {2, 3, 4}, {2, 3, 4});
  assertNumpyBroadcast({1, 1, 1}, {2, 3, 4}, {2, 3, 4});
  assertNumpyBroadcast({3, 4}, {2, 3, 4}, {2, 3, 4});
  assertNumpyBroadcast({3, 1}, {2, 3, 4}, {2, 3, 4});
  assertNumpyBroadcast({1, 1}, {2, 3, 4}, {2, 3, 4});
  assertNumpyBroadcast({2, 3, 4}, {1}, {2, 3, 4});
}

void testRowMajorIndex0() {

  // For shape {2,3,4}:
  // 0  0  0   :  0
  // 0  0  1   :  1
  // 0  0  2   :  2
  // 0  0  3   :  3
  // 0  1  0   :  4
  // 0  1  1   :  5
  // 0  1  2   :  6
  // 0  1  3   :  7
  // 0  2  0   :  8
  // 0  2  1   :  9
  // 0  2  2   :  10
  // 0  2  3   :  11
  // 1  0  0   :  12
  // 1  0  1   :  13
  // 1  0  2   :  14
  // 1  0  3   :  15
  // 1  1  0   :  16
  // 1  1  1   :  17
  // 1  1  2   :  18
  // 1  1  3   :  19
  // 1  2  0   :  20
  // 1  2  1   :  21
  // 1  2  2   :  22
  // 1  2  3   :  23
  // etc.
  assertRowMajorIndex(Shape({2, 3, 4}), {0, 2, 2}, 10);
  assertRowMajorIndex(Shape({2, 3, 4}), {1, 0, 3}, 15);

  assertRowMajorIndex(Shape({2, 3, 5, 1, 1}), {0, 2, 1, 0, 0}, 11);
}
} // namespace

int main() {
  testNumpyBinary0();
  testRowMajorIndex0();
  return 0;
}
