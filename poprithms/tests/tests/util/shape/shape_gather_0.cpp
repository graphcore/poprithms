// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/ndarray/error.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace {
using namespace poprithms::ndarray;

void confirmRowMajorIndices(const Shape &s,
                            uint64_t dimension,
                            const std::vector<int64_t> &where,
                            const std::vector<int64_t> &expected) {

  auto obs = s.gatherRowMajorIndices(dimension, where);
  if (obs != expected) {
    std::ostringstream oss;
    oss << "Expected " << expected << ", but observed " << obs
        << " in confirmRowMajorIndices, where dimension = " << dimension
        << " and where = " << where;
    throw error(oss.str());
  }
}

void testGather0() {

  auto expected = [](int64_t i, int64_t j, int64_t k) {
    return 15 * i + 5 * j + k;
  };

  Shape s({2, 3, 5});
  const std::vector<int64_t> I{1};
  const std::vector<int64_t> J{1, 2};
  const std::vector<int64_t> K{0, 2, 3};
  std::vector<int64_t> E;
  for (auto i : I) {
    for (auto j : J) {
      for (auto k : K) {
        E.push_back(expected(i, j, k));
      }
    }
  }

  const auto gathered = s.gatherRowMajorIndices({I, J, K});
  if (gathered != E) {
    std::ostringstream oss;
    oss << "Failure in multi-dimensionsal gather test. "
        << "Expected " << E << ", observed " << gathered << '.';
    throw error(oss.str());
  }
}
} // namespace

int main() {
  confirmRowMajorIndices({3, 3}, 0, {0, 2}, {0, 1, 2, 6, 7, 8});
  confirmRowMajorIndices({3, 3}, 1, {0, 2}, {0, 2, 3, 5, 6, 8});
  confirmRowMajorIndices({2, 3}, 1, {0, 2, 0}, {0, 2, 0, 3, 5, 3});
  confirmRowMajorIndices({2, 5, 3}, 1, {0}, {0, 1, 2, 15, 16, 17});
  confirmRowMajorIndices(
      {2, 5, 3}, 1, {2, 0}, {6, 7, 8, 0, 1, 2, 21, 22, 23, 15, 16, 17});

  testGather0();
}
