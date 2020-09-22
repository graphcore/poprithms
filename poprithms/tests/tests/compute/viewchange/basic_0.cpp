// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/viewchange.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::compute::host;

void expandTest0() {
  std::vector<int> iotic(15);
  std::iota(iotic.begin(), iotic.end(), 0);
  const auto out =
      ViewChange<int>::expand({{3, 1, 5}, iotic.data()}, {3, 4, 5});

  // 01234 01234 01234 01234
  // 56789 56789 56789 56789
  std::vector<int> expected(60);
  for (uint64_t i = 0; i < 60; ++i) {
    expected[i] = i % 5 + 5 * (i / 20);
  }
  if (out != expected) {
    throw error("Unexpected result in basic expand test");
  }
}

void dimShuffleTest0() {
  std::vector<int> iotic(2 * 2 * 2);
  std::iota(iotic.begin(), iotic.end(), 0);
  const auto out =
      ViewChange<int>::dimShuffle({{2, 2, 2}, iotic.data()}, {{2, 0, 1}});
  const std::vector<int> expected{0, 2, 4, 6, 1, 3, 5, 7};
  if (expected != out) {
    throw error("Unexpected result in basic dim shuffle test");
  }
}

void sliceTest0() {
  std::vector<int> iotic(15);
  std::iota(iotic.begin(), iotic.end(), 0);
  const auto out =
      ViewChange<int>::slice({{3, 5}, iotic.data()}, {0, 0}, {2, 3});
  const std::vector<int> expected{0, 1, 2, 5, 6, 7};
  if (expected != out) {
    throw error("Unexpected result in basic slice test");
  }
}

void concatTest0() {
  const std::vector<std::vector<uint64_t>> toConcat{
      {0, 1, 2}, {3, 4, 5, 6, 7, 8}, {9, 10, 11}};
  const auto a = ViewChange<uint64_t>::concat(
      {toConcat[0].data(), toConcat[1].data()}, {{1, 3}, {2, 3}}, 0);
  const auto b = ViewChange<uint64_t>::concat(
      {a.data(), toConcat[2].data()}, {{3, 3}, {3, 1}}, 1);
  const std::vector<uint64_t> expected{0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8, 11};
  if (b != expected) {
    std::ostringstream oss;
    oss << "Failure on concat test, expected:\n"
        << "       0 1 2 9\n"
        << "       3 4 5 10\n"
        << "       6 7 8 11\n\nnot:";
    poprithms::util::append(oss, b);
    throw error(oss.str());
  }
}

} // namespace

int main() {
  expandTest0();
  dimShuffleTest0();
  sliceTest0();
  concatTest0();
  return 0;
}
