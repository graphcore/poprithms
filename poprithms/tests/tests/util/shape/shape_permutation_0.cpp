// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using poprithms::ndarray::Shape;
using poprithms::util::Permutation;

void assertPermutation(const Shape &inshape,
                       const Shape &outshape,
                       const Permutation &perm,
                       const Permutation &expected) {
  const auto observed = inshape.moveDimShuffleFirst(outshape, perm);

  if (observed.second != expected) {
    std::ostringstream oss;
    oss << "Test of moving Permutation backwards failed. Initial \"network\" "
           "is\n"
        << inshape << " --reshape--> " << outshape << "--perm " << perm
        << "-->. \n"
        << "expected this to be transformed to \n"
        << inshape << "--perm " << expected << "-->, but instead of "
        << expected << ", " << observed.second << " was computed. ";
    throw poprithms::test::error(oss.str());
  }
}
void test0() {

  /*
   * original:
   * (2,3,5) -> (6,5) -> (1 0) -> (5,6)
   *
   *
   * 2   3 5   inShape
   *  \ /  |      |
   *   6   5   outShape
   *              |
   *        dimShuffle (1 0)
   *
   *
   * becomes:
   * (2,3,5) -> (2 0 1) -> (5,2,3) -> (5,6)
   *             ======
   *
   * */
  assertPermutation({2, 3, 5}, {6, 5}, {{1, 0}}, {{2, 0, 1}});

  assertPermutation({2, 3, 5, 7}, {6, 35}, {{1, 0}}, {{2, 3, 0, 1}});

  assertPermutation({6, 5}, {2, 3, 5}, {{2, 0, 1}}, {{1, 0}});

  assertPermutation({6, 35}, {2, 3, 5, 7}, {{2, 3, 0, 1}}, {{1, 0}});

  /*    0    1   2     3    4   5   6
   *
   *     2   3   35    12   2   3   100
   *      \ /   / \    / \   \ /    /  \
   *       6   5   7  3   4   6    10  10
   *
   *       0   1   2  3   4   5    6   7
   *      ===  =====  =====  ===   =====
   *      0,11   2      3    4,5    100
   */
  assertPermutation({2, 3, 35, 12, 2, 3, 100},
                    {6, 5, 7, 3, 4, 6, 10, 10},
                    {{5, 6, 7, 1, 2, 0, 3, 4}},
                    {{4, 5, 6, 2, 0, 1, 3}});

  // when target shape has 1.
  assertPermutation({2, 3}, {1, 1, 1, 6, 1}, {{4, 3, 2, 1, 0}}, {{0, 1}});
  assertPermutation(
      {2, 3, 4, 5}, {6, 1, 20, 1}, {{2, 0, 3, 1}}, {{2, 3, 0, 1}});

  // when source shape has 1.
}

void assertNotPossible(const Shape &inShape,
                       const Shape &reshape,
                       const Permutation &perm) {
  std::cout << "with inShape = " << inShape << std::endl;
  const auto x = inShape.moveDimShuffleFirst(reshape, perm);
  if (x.first) {
    std::ostringstream oss;
    oss << "Attempt to move Permutation " << perm
        << " before the reshape (from " << inShape << " to " << reshape
        << ") passed, but should not have. ";
    throw poprithms::test::error(oss.str());
  }
}

void test1() {
  assertNotPossible({6}, {2, 3}, {{1, 0}});
  assertNotPossible({5, 6}, {5, 2, 3}, {{0, 2, 1}});

  assertNotPossible({1, 5, 6}, {5, 2, 3}, {{0, 2, 1}});
  assertNotPossible({5, 6}, {5, 2, 3}, {{0, 2, 1}});
  assertNotPossible({6, 5}, {5, 6}, {{0, 1}});
  assertNotPossible({6, 1, 5}, {5, 6}, {{0, 1}});
  assertNotPossible({35, 2, 3}, {5, 7, 6}, {{1, 0, 2}});
  assertNotPossible({35, 2, 3}, {5, 7, 6}, {{0, 2, 1}});
  assertNotPossible({35, 2, 3, 1}, {5, 7, 6}, {{0, 2, 1}});
  assertNotPossible({16}, {2, 2, 2, 2}, {{2, 3, 0, 1}});
  assertNotPossible({16}, {2, 2, 4}, {{0, 2, 1}});
  assertNotPossible({16, 1, 1, 1, 1}, {2, 2, 4}, {{0, 2, 1}});
  assertNotPossible({16, 1, 1, 1, 1}, {2, 2, 4, 1, 1}, {{0, 2, 1, 3, 4}});
}

void testWithInOnes(const Shape &inShape,
                    const Shape &outShape,
                    const Permutation &perm0,
                    const std::vector<uint64_t> &expectedSub) {

  const auto observed = inShape.moveDimShuffleFirst(outShape, perm0);

  // solution must have ..4..5..2..3 in that order. It doesn't matter where 0
  // and 3 go.
  if (observed.second.size() != inShape.rank_u64() ||
      !observed.second.subPermutation(expectedSub).isIdentity()) {
    std::ostringstream oss;
    oss << "Test of moveDimShuffleFirst where inShape has 1's. "
        << "Expected solution to of size " << inShape.rank_u64()
        << " and contain ";
    poprithms::util::append(oss, expectedSub);
    oss << " in sequence "
        << " but " << observed.second << " does not fit this pattern. ";
    throw poprithms::test::error(oss.str());
  }
}

void testWithInOnes0() {
  Shape inShape({1, 2, 3, 1, 4, 5});
  Shape outShape({6, 1, 20, 1});
  Permutation perm0({2, 0, 3, 1}); // 20, 6, 1, 1
  testWithInOnes(inShape, outShape, perm0, {4, 5, 2, 3});

  //      1   6  1  4  1  5      in
  //        /  \     \   /.
  //      2  1  3     20   1 1  out
  //
  // Permutation on out produces (1, 20, 1, 1, 2,3)
  // So if the Permutation is done on the input Shape before the reshape, it
  // must put 4 before 5 before 6 (hence 3,5,1 expectation).
  //
  testWithInOnes({1, 6, 1, 4, 1, 5},
                 {2, 1, 3, 20, 1, 1},
                 {{1, 3, 4, 5, 0, 2}},
                 {3, 5, 1});
}

} // namespace

int main() {
  test0();
  test1();
  testWithInOnes0();
  return 0;
}
