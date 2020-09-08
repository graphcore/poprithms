// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/shape.hpp>

namespace {

using namespace poprithms::util;

void assertDimShuffleIndices(const Shape &s,
                             const Permutation &p,
                             const std::vector<int64_t> &expected) {

  const auto indices = s.getDimShuffledRowMajorIndices(p);
  if (indices != expected) {
    std::ostringstream oss;
    oss << "Error in assertDimShuffleIndices. Expected\n   ";
    poprithms::util::append(oss, expected);
    oss << ", \nbut observed\n   ";
    poprithms::util::append(oss, indices);
    throw error(oss.str());
  }
}

void assertExpandedIndices(const Shape &from,
                           const Shape &to,
                           const std::vector<int64_t> &expected) {

  const auto indices = from.getExpandedRowMajorIndices(to);
  if (indices != expected) {
    std::ostringstream oss;
    oss << "Error in assertDimShuffleIndices. Expected\n   ";
    poprithms::util::append(oss, expected);
    oss << ", \nbut observed\n   ";
    poprithms::util::append(oss, indices);
    throw error(oss.str());
  }
}
} // namespace

int main() {

  const Shape s({2, 3});
  const Permutation p{{1, 0}};
  assertDimShuffleIndices(s, p, {0, 3, 1, 4, 2, 5});

  //  [[[ 0 1 ]
  //    [ 2 3 ]]
  //   [[ 4 5 ]
  //    [ 6 7 ]]]

  // 0 is fastest changing, then 2, and 1 is slowest changing.
  assertDimShuffleIndices({2, 2, 2}, {{1, 2, 0}}, {0, 4, 1, 5, 2, 6, 3, 7});

  // 1 is fastest changing, then 0, and 2 is slowest changing.
  assertDimShuffleIndices({2, 2, 2}, {{2, 0, 1}}, {0, 2, 4, 6, 1, 3, 5, 7});

  // [[[[0]
  //    [1]]
  //   [[2]
  //    [3]]
  //   [[4]
  //    [5]]]]
  assertDimShuffleIndices({1, 2, 3, 1}, {{3, 2, 1, 0}}, {0, 3, 1, 4, 2, 5});

  assertExpandedIndices({3, 1}, {3, 2}, {0, 0, 1, 1, 2, 2});

  assertExpandedIndices({1, 3}, {2, 3}, {0, 1, 2, 0, 1, 2});
  assertExpandedIndices({2, 1, 3}, {2, 4, 3}, {0, 1, 2, 0, 1, 2, //
                                               0, 1, 2, 0, 1, 2, //
                                               3, 4, 5, 3, 4, 5, //
                                               3, 4, 5, 3, 4, 5});
  assertExpandedIndices({2}, {4, 2}, {0, 1, 0, 1, 0, 1, 0, 1});
  assertExpandedIndices({}, {1, 2, 1}, {0, 0});

  const auto s1 = Shape({3, 2});
  auto strides  = s1.getCustomStridedRowMajorIndices({4, 4});
  if (strides != decltype(strides){0, 4, 4, 8, 8, 12}) {
    throw error("Error in generalized method, getRowMajorIndices");
  }

  return 0;
}
