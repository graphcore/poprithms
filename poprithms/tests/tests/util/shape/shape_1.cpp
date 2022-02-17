// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::ndarray;
void testCreateFrom0() {

  auto s0 = Shape::createFrom(std::vector<uint32_t>{1, 2, 3});
  auto s1 = Shape::createFrom(std::array<int8_t, 3>{1, 2, 3});

  std::vector<size_t> countUp{1ull, 2ull, 3ull};

  auto s2 = Shape::createFrom(countUp);
  auto s3 = Shape::createFrom(std::move(countUp));

  for (Shape x : {s0, s1, s2, s2}) {
    if (x != Shape({1, 2, 3})) {
      throw poprithms::test::error(
          "Incorrect construction of Shape using createFrom");
    }
  }
}

void testFlatten0() {
  {

    Shape s{};
    auto x = s.flatten(0, 0);
    x      = x.flatten(1, 1);
    x      = x.flatten(1, 1);
    if (x != Shape({1, 1, 1})) {
      throw poprithms::test::error(
          "flatten with from=to is equivalent to unsqueeze");
    }
  }

  auto x = Shape({})
               .flatten(0, 0)
               .flatten(1, 1)
               .flatten(2, 2)
               .flatten(0, 0)
               .flatten(1, 1);
  if (x != Shape({1, 1, 1, 1, 1})) {
    throw poprithms::test::error(
        "Four flattens with from=to, expected (1,1,1,1,1)");
  }
}

void testFlatten1() {
  {
    const auto s = Shape({7}).flatten(0, 1).flatten(1, 1).flatten(1, 2);
    if (s != Shape({7, 1})) {
      throw poprithms::test::error(
          "Expected the flatten with from=to=1 to put a 1 on the end");
    }
  }

  {
    auto s = Shape({2, 3, 5, 7, 11});
    s      = s.flatten(1, 3);               // 2 15 7 11
    s      = s.flatten(2, 4);               // 2 15 77
    s      = s.flatten(0, 2);               // 30, 77
    s      = s.flatten(1, 2).flatten(1, 1); // 30, 1, 77
    if (s != Shape({30, 1, 77})) {
      throw poprithms::test::error(
          "Expected this chain of flattens to produce 33,1,77");
    }
  }
}

void validateInsertOnesAt(const Shape &inShape,
                          const std::vector<uint64_t> &dims,
                          const Shape &expected) {
  if (inShape.insertOnesAt(dims) != expected) {
    std::ostringstream oss;
    oss << "Expected " << expected << " with " << inShape << ".insertOnesAt(";
    poprithms::util::append(oss, dims);
    oss << ") but observed " << inShape.insertOnesAt(dims) << '.';
    throw poprithms::test::error(oss.str());
  }
}
void testInsertOnesAt0() {
  validateInsertOnesAt(
      Shape{}, /* dims */ {0, 0}, /* expected */ Shape{1, 1});

  validateInsertOnesAt({}, {}, {});
  validateInsertOnesAt({2, 3, 5}, {}, {2, 3, 5});
  validateInsertOnesAt({2, 4, 5}, {}, {2, 4, 5});
  validateInsertOnesAt({3, 4}, {0, 0}, {1, 1, 3, 4});
  validateInsertOnesAt({3, 4}, {1, 1}, {3, 1, 1, 4});
  validateInsertOnesAt({3, 4}, {2, 2}, {3, 4, 1, 1});
  validateInsertOnesAt({3, 4}, {2, 1, 0}, {1, 3, 1, 4, 1});

  bool caught{false};
  try {
    Shape({}).insertOnesAt({1});
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch invalid dimension in insertOnesAt");
  }
}

} // namespace

int main() {
  testCreateFrom0();

  testFlatten0();
  testFlatten1();
  testInsertOnesAt0();

  return 0;
}
