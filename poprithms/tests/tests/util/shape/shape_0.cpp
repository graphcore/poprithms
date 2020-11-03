// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/ndarray/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::ndarray;

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

void testConcat() {
  const Shape a({2, 3, 4});
  const Shape b({2, 2, 4});
  const auto c = a.concat(b, 1);
  if (c != Shape({2, 5, 4})) {
    throw error("Failed in testConcat (1)");
  }

  const Shape d({0, 3, 4});
  const auto e = d.concat(a, 0);
  if (e != a) {
    throw error("Failed in testConcat (2)");
  }

  auto points = Shape::concatPartitionPoints({a, b}, 1);
  if (points != std::vector<int64_t>({0, 3, 5})) {
    throw error("Failed in testConcat : incorrect partition points");
  }
}

void testSqueeze() {
  const Shape a({2, 1, 1, 3, 1, 1, 4});
  auto s = a.squeeze();
  if (s != Shape({2, 3, 4})) {
    throw error(
        "Failed to squeeze correctly in testSqueeze, expected (2,3,4)");
  }

  const auto foo = s.unsqueeze(0);
  if (foo != Shape({1, 2, 3, 4})) {
    throw error(
        "Failed to unsqueeze correctly in testSqueeze, expected (1,2,3,4)");
  }
}

void testSqueeze2() {
  //            x           x
  const Shape s{1, 2, 1, 3, 1};
  const std::vector<uint64_t> dims{4, 0, 0, 0, 0, 0, 0, 0};
  const auto out = s.squeeze(dims);
  if (out != Shape{2, 1, 3}) {
    throw error("Failed in testUnsqueeze2, expected {2,1,3}");
  }
}

void assertDimProduct(uint64_t x0, uint64_t x1, int64_t expected) {
  const Shape s0({4, 3});
  if (s0.dimProduct(x0, x1) != expected) {
    std::ostringstream oss;
    oss << "Failure in assertDimProduct, in test of Shape class. "
        << "For Shape " << s0 << " in call to dimProduct(" << x0 << ", " << x1
        << "), expected is " << expected << ". Observed is "
        << s0.dimProduct(x0, x1);
    throw error(oss.str());
  }
}

void testDimProduct() {
  assertDimProduct(0, 1, 4);
  assertDimProduct(1, 2, 3);
  assertDimProduct(0, 2, 12);
  assertDimProduct(0, 0, 1);
  assertDimProduct(1, 1, 1);
}

void testReverse() {
  const Shape s1({1, 2, 3});
  if (s1.reverse() != Shape{{3, 2, 1}}) {
    throw error("Error in test of reverse");
  }
}

void testGetRowMajorIndices() {
  const Shape s1({4, 3});
  const auto inds = s1.getSlicedRowMajorIndices({1, 1}, {3, 2});
  if (inds != decltype(inds){4, 7}) {
    throw error("Error in test of get row major indices (from slice)");
  }
}

void testPrepend() {
  Shape s0({});
  s0 = s0.prepend(4);
  s0 = s0.prepend(3);
  s0 = s0.prepend(2);
  if (s0 != Shape{2, 3, 4}) {
    std::ostringstream oss;
    oss << "Failure in prepend test. Result is " << s0
        << ". Prepending 4, then 3, then 2, should produce (2,3,4).";
    throw error(oss.str());
  }
}

void assertUnsqueeze(const Shape &a,
                     const std::vector<uint64_t> &dims,
                     const Shape &b) {
  if (a.unsqueeze(dims) != b) {
    std::ostringstream oss;
    oss << "Error in testing unsqueeze. Expected " << a << ".unsqueeze(";
    poprithms::util::append(oss, dims);
    oss << ") to be " << b << ", not " << a.unsqueeze(dims);
    throw error(oss.str());
  }
}

void testUnsqueeze0() {
  assertUnsqueeze({}, {}, {});
  assertUnsqueeze({}, {0, 1}, {1, 1});
  assertUnsqueeze({2, 3}, {}, {2, 3});
  assertUnsqueeze({2, 3}, {0}, {1, 2, 3});
  assertUnsqueeze({2, 3}, {0, 3}, {1, 2, 3, 1});
  assertUnsqueeze({2, 3}, {0, 2, 4}, {1, 2, 1, 3, 1});
  assertUnsqueeze({2, 3}, {0, 4, 3}, {1, 2, 3, 1, 1});
}

void testPadShapes0() {
  const auto padShapes = Shape({1, 2}).getPadShapes({0, 1}, {2, 3});

  // Where "xx" is the Shape being padded:
  //
  //  100111
  //  100111
  //  1xx111.
  //
  std::vector<std::array<Shape, 2>> expected;
  expected.push_back({{{0, 2}, {2, 2}}});
  expected.push_back({{{3, 1}, {3, 3}}});
  if (padShapes != expected) {
    std::ostringstream oss;
    oss << "Unexpected result in testPadShapes ";
    oss << '(';
    for (auto padShape : padShapes) {
      poprithms::util::append(oss, std::get<0>(padShape).get());
      poprithms::util::append(oss, std::get<1>(padShape).get());
    }
    oss << ')';
    throw error(oss.str());
  }
}

void assertFlatten2d(const Shape &inShape, uint64_t axis, const Shape &out) {
  if (inShape.flattenTo2d(axis) != out) {
    std::ostringstream oss;
    oss << "Error in assertFlatten2d. Expected " << inShape << ".flattenTo2d("
        << axis << ") to be " << out << ", not " << inShape.flattenTo2d(axis)
        << ".";
    throw error(oss.str());
  }
}

void testFlatten2d() {
  assertFlatten2d({2, 3, 4}, 0, {1, 24});
  assertFlatten2d({2, 3, 4}, 1, {2, 12});
  assertFlatten2d({2, 3, 4}, 2, {6, 4});
  assertFlatten2d({2, 3, 4}, 3, {24, 1});
}

int main() {
  testNumpyBinary0();
  testPrepend();
  testRowMajorIndex0();
  testConcat();
  testSqueeze();
  testSqueeze2();
  testDimProduct();
  testReverse();
  testGetRowMajorIndices();
  testUnsqueeze0();
  testPadShapes0();
  testFlatten2d();
  return 0;
}
