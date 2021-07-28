// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
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

} // namespace

int main() {
  testCreateFrom0();

  return 0;
}
