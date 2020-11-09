// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
} // namespace

int main() {
  confirmRowMajorIndices({3, 3}, 0, {0, 2}, {0, 1, 2, 6, 7, 8});
  confirmRowMajorIndices({3, 3}, 1, {0, 2}, {0, 2, 3, 5, 6, 8});
  confirmRowMajorIndices({2, 3}, 1, {0, 2, 0}, {0, 2, 0, 3, 5, 3});
  confirmRowMajorIndices({2, 5, 3}, 1, {0}, {0, 1, 2, 15, 16, 17});
  confirmRowMajorIndices(
      {2, 5, 3}, 1, {2, 0}, {6, 7, 8, 0, 1, 2, 21, 22, 23, 15, 16, 17});
}
