// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/viewchange.hpp>
#include <poprithms/error/error.hpp>
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
    throw poprithms::test::error("Unexpected result in basic expand test");
  }
}

void dimShuffleTest0() {
  std::vector<int> iotic(2 * 2 * 2);
  std::iota(iotic.begin(), iotic.end(), 0);
  const auto out =
      ViewChange<int>::dimShuffle({{2, 2, 2}, iotic.data()}, {{2, 0, 1}});
  const std::vector<int> expected{0, 2, 4, 6, 1, 3, 5, 7};
  if (expected != out) {
    throw poprithms::test::error(
        "Unexpected result in basic dim shuffle test");
  }
}

void sliceTest0() {
  std::vector<int> iotic(15);
  std::iota(iotic.begin(), iotic.end(), 0);
  const auto out =
      ViewChange<int>::slice({{3, 5}, iotic.data()}, {0, 0}, {2, 3});
  const std::vector<int> expected{0, 1, 2, 5, 6, 7};
  if (expected != out) {
    throw poprithms::test::error("Unexpected result in basic slice test");
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
    throw poprithms::test::error(oss.str());
  }
}

void reverseTest(const std::vector<int> &input,
                 const std::vector<int> &expectedOut,
                 const Shape &shape,
                 const std::vector<uint64_t> &dims) {
  const auto out = ViewChange<int>::reverse({shape, input.data()}, dims);
  if (out != expectedOut) {
    std::ostringstream oss;
    oss << "Error detected in reverseTest. "
        << "This with \ninput=";
    poprithms::util::append(oss, input);
    oss << "\nobserved output=";
    poprithms::util::append(oss, out);
    oss << "\nexpected output=";
    poprithms::util::append(oss, expectedOut);
    oss << "\nShape=" << shape << "\ndimensions=";
    poprithms::util::append(oss, dims);
    throw poprithms::test::error(oss.str());
  }
}

void reverseTest0() {
  reverseTest({0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, Shape{2, 3}, {0, 1});
  reverseTest({0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, Shape{2, 3}, {0, 0});
  reverseTest({0, 1, 2, 3, 4, 5}, {3, 4, 5, 0, 1, 2}, Shape{2, 3}, {0});
  reverseTest({0, 1, 2, 3, 4, 5}, {2, 1, 0, 5, 4, 3}, Shape{2, 3}, {1});
  reverseTest({0, 1, 2, 3, 4, 5, 6, 7},
              {1, 0, 3, 2, 5, 4, 7, 6},
              Shape{2, 2, 2},
              {2});
  reverseTest({0, 1, 2, 3, 4, 5, 6, 7},
              {5, 4, 7, 6, 1, 0, 3, 2},
              Shape{2, 2, 2},
              {0, 2});
}

void subsampleTest(const std::vector<int> &input,
                   const std::vector<int> &expectedOut,
                   const Shape &inShape,
                   const std::vector<uint64_t> &strides) {
  const auto out =
      ViewChange<int>::subSample({inShape, input.data()}, strides);
  if (out != expectedOut) {
    std::ostringstream oss;
    oss << "Error detected in reverseTest. "
        << "This with \ninput=";
    poprithms::util::append(oss, input);
    oss << "\nobserved output=";
    poprithms::util::append(oss, out);
    oss << "\nexpected output=";
    poprithms::util::append(oss, expectedOut);
    oss << "\ninShape=" << inShape << "\nstrides=";
    poprithms::util::append(oss, strides);
    throw poprithms::test::error(oss.str());
  }
}

void subSampleTest0() {
  subsampleTest({0, 1, 2, 3}, {0, 1, 2, 3}, Shape{2, 2}, {1, 1});
  subsampleTest({0, 1, 2, 3}, {0}, Shape{2, 2}, {2, 2});
  subsampleTest({0, 1, 2, 3}, {0, 1}, Shape{2, 2}, {2, 1});
  subsampleTest({0, 1, 2, 3}, {0, 2}, Shape{2, 2}, {1, 2});
  subsampleTest(
      {0, 1, 2, 3, 4, 5}, {0}, Shape{1, 3, 2, 1, 1}, {10, 11, 12, 13, 14});
}

} // namespace

int main() {
  expandTest0();
  dimShuffleTest0();
  sliceTest0();
  concatTest0();
  reverseTest0();
  subSampleTest0();
  return 0;
}
