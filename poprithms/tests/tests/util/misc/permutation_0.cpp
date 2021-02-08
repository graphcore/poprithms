// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>

namespace {

using namespace poprithms::util;

void test0() {
  Permutation p({1, 2, 0, 4, 5, 3});
  const auto inv = p.inverse();
  if (inv != Permutation({2, 0, 1, 5, 3, 4})) {
    throw error("Unexpected inverse in Permutation test");
  }
  if (inv.isIdentity()) {
    throw error("This Permutation is not identity, test failure");
  }
  const auto permuted = p.apply(std::vector<int>{13, 11, 7, 5, 3, 2});
  if (permuted != std::vector<int>{11, 7, 13, 3, 2, 5}) {
    throw error("Permuted vector is not as expected");
  }
}

void testProd0() {
  // A cycle:

  Permutation p0({1, 2, 3, 0});
  const auto x4 = Permutation::prod(std::vector<Permutation>(4, p0));
  if (!x4.isIdentity()) {
    throw error("A Permutation of size 4, applied to itself 4 times, is "
                "always identity");
  }

  const auto x2 = Permutation::prod(std::vector<Permutation>(2, p0));
  if (x2 != Permutation({2, 3, 0, 1})) {
    throw error("Expected (1 2 3 0) o (1 2 3 0) == (2 3 0 1)");
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
    throw error(oss.str());
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
} // namespace

int main() {
  test0();
  testProd0();
  testDimRoll0();
  return 0;
}
