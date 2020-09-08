// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/shape.hpp>

namespace {

using namespace poprithms::util;

void assertBlockOrder(const Shape &shape,
                      const Shape &block,
                      const std::vector<int64_t> &expected) {

  const auto indices = shape.getRowMajorBlockOrdered(block);
  if (indices != expected) {
    std::ostringstream oss;
    oss << "Error in assertBlockOrder. Expected\n   ";
    poprithms::util::append(oss, expected);
    oss << ", \nbut observed\n   ";
    poprithms::util::append(oss, indices);
    throw error(oss.str());
  }
}

} // namespace

int main() {

  assertBlockOrder({5, 5},
                   {2, 3},
                   {
                       0,  1,  2,  5,  6,  7,  //
                       3,  4,  8,  9,          //
                       10, 11, 12, 15, 16, 17, //
                       13, 14, 18, 19,         //
                       20, 21, 22, 23, 24      //
                   });

  std::vector<int64_t> expected;

  auto indices = Shape({4, 8, 7}).getRowMajorBlockOrdered({3, 2, 5});
  std::sort(indices.begin(), indices.end());
  for (int64_t i = 0; i < 4 * 8 * 7; ++i) {
    if (indices[i] != i) {
      throw error("Failed in check for all indices (block order)");
    }
  }

  return 0;
}
