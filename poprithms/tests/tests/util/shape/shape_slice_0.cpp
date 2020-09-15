// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/ndarray/error.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace {
using namespace poprithms::ndarray;
using Lower = Shape::Lower;
using Upper = Shape::Upper;

void testSlice() {
  Shape f({4, 5});
  auto o = f.slice({0, 1}, {2, 3});
  if (o != Shape({2, 2})) {
    throw error("Failed in testSlice");
  }
}

void confirmRowMajorIndices(const Shape &s,
                            const Lower &l,
                            const Upper &u,
                            const std::vector<int64_t> &expected) {

  std::cout << "In confirmRowMajorIndices" << std::endl;

  auto obs = s.getSlicedRowMajorIndices(l, u);
  if (obs != expected) {
    std::ostringstream oss;
    oss << "Expected " << expected << ", but observed " << obs
        << " in confirmRowMajorIndices, where s = " << s << ", l = " << l
        << ", and u = " << u;
    throw error(oss.str());
  }
}
} // namespace

int main() {
  testSlice();

  //{15} + {5} x {1,2}
  confirmRowMajorIndices({2, 3, 5}, {1, 1, 1}, {2, 2, 3}, {21, 22});

  //{15} x {5,10} x {1,2}
  confirmRowMajorIndices({2, 3, 5}, {1, 1, 1}, {2, 3, 3}, {21, 22, 26, 27});

  // {0, 15} x  {10} x {2}
  confirmRowMajorIndices({2, 3, 5}, {0, 2, 2}, {2, 3, 3}, {12, 27});
}
