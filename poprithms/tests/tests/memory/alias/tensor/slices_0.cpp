// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void testSlices0() {
  Graph g;
  auto tensor = g.tensor(g.allocate({50}));

  std::vector<Interval> slices;
  for (int64_t i = 0; i < 10; ++i) {
    slices.push_back({5 * i, 5 * (i + 1)});
  }

  auto sliced = tensor.slices(slices, 0);
  for (int64_t i = 0; i < 10; ++i) {
    for (int64_t j = 0; j < 10; ++j) {
      if (j == i)
        continue;
      if (sliced[j].intersectsWith(sliced[i])) {
        std::ostringstream oss;
        oss << "Test failure: slices[" << j
            << "] should not intersect with slices[" << i << "]";
        throw poprithms::test::error(oss.str());
      }
    }
  }

  bool caught{false};
  try {
    auto slicedError = tensor.slices(slices, 1);
  } catch (poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Unexpected passing test: slice dimension "
                                 "shouldn't exceed tensor rank.");
  }

  std::vector<Interval> emptySlices;
  auto slicedEmpty = tensor.slices(emptySlices, 0);
  if (!slicedEmpty.empty()) {
    throw poprithms::test::error(
        "Empty sequence of slice intervals should result in no tensors.");
  }
}

void testSlices1() {
  Graph g;
  auto tensor = g.tensor(g.allocate({50}));

  std::vector<std::vector<Interval>> sliceSequences;
  std::vector<Interval> oddSlices;
  std::vector<Interval> evenSlices;
  for (uint64_t i = 0; i < 50; ++i) {
    if (i & 1) {
      oddSlices.push_back({i, i + 1});
    } else {
      evenSlices.push_back({i, i + 1});
    }
  }

  sliceSequences.push_back(std::move(oddSlices));
  sliceSequences.push_back(std::move(evenSlices));

  // tensor         -> oddSlices
  // [_x_x...._x_x] -> [xx...xx]
  // shape: {50}    -> shape: {25}
  //    and
  // tensor         -> evenSlices
  // [x_x_....x_x_] -> [xx...xx]
  // shape: {50}    -> shape: {25}
  //
  // Since the odd or even slices themselves don't overlap, no intersection
  // expected between the resulting concatenated tensors.

  auto slices = tensor.slices(sliceSequences, 0);

  if (slices[0].intersectsWith(slices[1])) {
    throw poprithms::test::error(
        "Test failure: odd and even slices of a tensor shouldn't intersect");
  }

  if (slices[0].shape() != Shape({25}) || slices[1].shape() != Shape({25})) {
    throw poprithms::test::error(
        "Test failure: shape after slices not as expected");
  }

  bool caught{false};
  try {
    auto slicedError = tensor.slices(sliceSequences, 1);
  } catch (poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Unexpected passing test: slice dimension "
                                 "shouldn't exceed tensor rank.");
  }
}

int main() {
  testSlices0();
  testSlices1();
  return 0;
}
