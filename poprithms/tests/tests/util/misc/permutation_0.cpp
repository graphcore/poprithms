// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using poprithms::util::Permutation;

std::ostream &operator<<(std::ostream &ost, const std::vector<uint64_t> &v) {
  poprithms::util::append(ost, v);
  return ost;
}

void test0() {
  Permutation p({1, 2, 0, 4, 5, 3});
  const auto inv = p.inverse();
  if (inv != Permutation({2, 0, 1, 5, 3, 4})) {
    throw poprithms::test::error("Unexpected inverse in Permutation test");
  }
  if (inv.isIdentity()) {
    throw poprithms::test::error(
        "This Permutation is not identity, test failure");
  }
  const auto permuted = p.apply(std::vector<int>{13, 11, 7, 5, 3, 2});
  if (permuted != std::vector<int>{11, 7, 13, 3, 2, 5}) {
    throw poprithms::test::error("Permuted vector is not as expected");
  }
}

void testProd0() {
  // A cycle:

  Permutation p0({1, 2, 3, 0});
  const auto x4 = Permutation::prod(std::vector<Permutation>(4, p0));
  if (!x4.isIdentity()) {
    throw poprithms::test::error(
        "A Permutation of size 4, applied to itself 4 times, is "
        "always identity");
  }

  const auto x2 = Permutation::prod(std::vector<Permutation>(2, p0));
  if (x2 != Permutation({2, 3, 0, 1})) {
    throw poprithms::test::error(
        "Expected (1 2 3 0) o (1 2 3 0) == (2 3 0 1)");
  }
}

void testDimRoll(const uint64_t rnk,
                 const uint64_t from,
                 const uint64_t to,
                 const Permutation &expected) {
  const auto p = Permutation::dimRoll(rnk, {from, to});
  if (p != expected) {
    std::ostringstream oss;
    oss << "Failed in test of Permutation's dimRoll. "
        << "With rnk = " << rnk << ", from = " << from << ", to = " << to
        << ", and expected = " << expected << "observed " << p
        << ", but expected " << expected << '.';
    throw poprithms::test::error(oss.str());
  }
}

void testDimRoll0() {
  testDimRoll(3, 0, 2, {{1, 2, 0}});
  testDimRoll(3, 2, 0, {{2, 0, 1}});
  testDimRoll(3, 0, 0, {{0, 1, 2}});
  testDimRoll(3, 2, 2, {{0, 1, 2}});
  testDimRoll(3, 0, 1, {{1, 0, 2}});
  testDimRoll(3, 1, 0, {{1, 0, 2}});
}

void testDimShufflePartial(uint64_t rnk,
                           const std::vector<uint64_t> &src,
                           const std::vector<uint64_t> &dst,
                           const Permutation &expected) {
  const auto p = Permutation::dimShufflePartial(rnk, src, dst);
  if (p != expected) {
    std::ostringstream oss;
    oss << "Failed in test of Permutation's dimShufflePartial. With src = "
        << src << ", dst = " << dst << ", resulting in permutation = " << p
        << ", while expected = " << expected << ".";
    throw poprithms::test::error(oss.str());
  }
}

void testDimShufflePartial0() {
  testDimShufflePartial(5, {3, 4}, {1, 3}, {{0, 3, 1, 4, 2}});
  testDimShufflePartial(5, {4, 3}, {3, 1}, {{0, 3, 1, 4, 2}});
  testDimShufflePartial(
      5, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {{0, 1, 2, 3, 4}});
  testDimShufflePartial(5, {0, 4}, {4, 0}, {{4, 1, 2, 3, 0}});

  testDimShufflePartial(3, {0, 1, 2}, {2, 1, 0}, {{2, 1, 0}});
  testDimShufflePartial(3, {0, 2}, {2, 1}, {{1, 2, 0}});
  testDimShufflePartial(0, {}, {}, {{}});
  testDimShufflePartial(1, {}, {}, {{0}});
}

void testDimShufflePartialError(const uint64_t rnk,
                                const std::vector<uint64_t> &src,
                                const std::vector<uint64_t> &dst) {
  bool caught{false};
  try {
    Permutation::dimShufflePartial(rnk, src, dst);
  } catch (poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Test succeeded unexpectedly with bad dimShufflePartial args.");
  }
}

void testDimShufflePartial1() {
  testDimShufflePartialError(5, {1, 2}, {4, 3, 2});
  testDimShufflePartialError(5, {1, 2, 3}, {3, 2});
  testDimShufflePartialError(3, {0, 1, 5}, {0, 1, 2});
  testDimShufflePartialError(3, {0, 1, 2}, {0, 5, 1});
  testDimShufflePartialError(3, {0, 0, 1}, {0, 1, 2});
  testDimShufflePartialError(3, {0, 1, 2}, {0, 1, 1});
  testDimShufflePartialError(3, {1, 2, 3, 4}, {5, 6, 7, 8});
}

void runSubsequenceBase(const Permutation &p,
                        const std::vector<uint64_t> &where,
                        const Permutation &expected) {
  const Permutation observed = p.subPermutation(where);
  if (observed != expected) {
    std::ostringstream oss;
    oss << "Failure in runSubsequenceBase, where Permutation p = " << p
        << ", where = " << where << ", and expected = " << expected
        << ". The observed solution is " << observed << '.';
    throw poprithms::test::error(oss.str());
  }
}

void testSubsequence() {

  //     This is (4 2 5 1 3 0) and where is (0,4,5)
  //              =   =     =
  //                         -> (1 2 0)
  runSubsequenceBase({{4, 2, 5, 1, 3, 0}}, {0, 4, 5}, {{1, 2, 0}});
  runSubsequenceBase({{4, 2, 5, 1, 3, 0}}, {5, 4, 0}, {{1, 0, 2}});

  runSubsequenceBase({{1, 2, 0}}, {0, 2}, {{1, 0}});
  runSubsequenceBase({{2, 1, 0}}, {0, 2}, {{1, 0}});
  runSubsequenceBase({{2, 1, 3, 0}}, {0, 2, 3}, {{1, 2, 0}});

  //    This is  (4 6 0 5 2 1 3) and where is (2,3,5,6)
  //                =   = =   =
  //                        -> (3 2 0 1)
  runSubsequenceBase({{4, 6, 0, 5, 2, 1, 3}},
                     {
                         2,
                         3,
                         5,
                         6,
                     },
                     {{3, 2, 0, 1}});
  runSubsequenceBase({{1, 2, 0}}, {0, 1}, {{1, 0}});
  runSubsequenceBase({{1, 2, 0}}, {1, 0}, {{0, 1}});
}

void testContainsSubsequenceBase(const Permutation &p,
                                 const std::vector<uint64_t> &x,
                                 bool expected) {

  const auto observed = p.containsSubSequence(x);
  if (observed != expected) {
    std::ostringstream oss;
    oss << "Testing if " << p << " contains " << x
        << ", expected : " << (expected ? "YES" : "NO");
    throw poprithms::test::error(oss.str());
  }
}

void testContainsSubsequence0() {

  Permutation p({3, 5, 6, 1, 4, 2, 0});

  testContainsSubsequenceBase(p, {5, 6, 1}, true);
  testContainsSubsequenceBase(p, {5}, true);
  testContainsSubsequenceBase(p, {}, true);
  testContainsSubsequenceBase(p, {0}, true);
  testContainsSubsequenceBase(p, {4, 2, 0}, true);
  testContainsSubsequenceBase(p, p.get(), true);

  testContainsSubsequenceBase(p, {100}, false);
  testContainsSubsequenceBase(p, {0, 3}, false);
  testContainsSubsequenceBase(p, {3, 6}, false);
  testContainsSubsequenceBase(p, {0, 1}, false);
}

} // namespace

int main() {
  test0();
  testProd0();
  testDimRoll0();
  testDimShufflePartial0();
  testDimShufflePartial1();
  testSubsequence();
  testContainsSubsequence0();
  return 0;
}
