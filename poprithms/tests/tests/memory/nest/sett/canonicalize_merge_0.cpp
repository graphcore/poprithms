// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/optionalset.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <testutil/memory/nest/randomsett.hpp>

namespace {

using namespace poprithms::memory::nest;

void baseTest(const Sett &a, const Sett &b, bool expected) {
  auto c = Sett::merge(a, b);
  if (expected != c.full()) {
    std::ostringstream oss;
    oss << "Failure in test of merge. Sett::merge(" << a << ", " << b
        << ") = " << c << ", but expected = " << expected << ".";
    throw error(oss.str());
  }

  if (c.full()) {
    Sett::confirmDisjoint({a, b});
    c.first().confirmEquivalent(DisjointSetts({a, b}));
  }
}

void testMergeA0() {

  // ..x.x.x.x.....x.x.x.x.....x.x.x.x...
  // x...........x...........x...........
  // 0123456789012345678
  //
  Sett a{{{100, 50, 37}, {7, 5, 2}, {1, 1, 0}}};
  for (int64_t offset : {0, 10}) {
    Sett b{{{100, 50, 37}, {1, 11, offset}}};
    // DO expect a merge
    baseTest(a, b, true);
  }

  for (int64_t offset : {1, 9, 11}) {
    Sett b{{{100, 50, 37}, {1, 11, offset}}};
    // do NOT expect a merge
    baseTest(a, b, false);
  }
}

void testMergeA1() {

  // ..xxxxxxxxxxxxxxxx..........xxxxxxxxxxxxxxxx.......... (16, 10, 2)
  //  .xxxxxxx..xxxxxxx                                     (7, 2, 4)
  //   .xxxx.x                                              (4, 1, 1)
  //    .xx.x                                               (2, 1, 0)
  //
  // 0123456789012345678901234567890
  //                     |
  //                   ..xxxxxxx........ (7, 19, 20)
  //                     .xxxx.x         (4, 1, 1)
  //                      .xx.           (2, 1, 0)
  //
  Sett a = Sett{{{16, 10, 2}, {7, 2, 0}, {4, 1, 1}, {2, 1, 0}}};
  Sett b{{{7, 19, 20}, {4, 1, 1}, {2, 1, 0}}};
  baseTest(a, b, true);
}

void testMergeB0() {
  baseTest({{{{4, 5, 2}}}}, {{{{1, 8, 1}}}}, true);
  baseTest({{{{4, 5, 2}}}}, {{{{1, 8, 0}}}}, false);
  baseTest({{{{4, 5, 2}}}}, {{{{1, 8, 2}}}}, false);
  baseTest({{{{4, 5, 2}}}}, {{{{1, 8, 6}}}}, true);
}

void testMergeC0() {
  baseTest({{{{1, 5, 2}}}}, {{{{1, 5, 0}}}}, true);
  baseTest({{{{2, 11, 4}}}}, {{{{2, 11, 7}}}}, true);

  const Stripe s0{117, 25, 9};
  const Stripe s1{2, 1, 1};
  baseTest({{{s0, {5, 11, 1}, s1}}}, {{{s0, {5, 11, 7}, s1}}}, true);
  baseTest({{{s0, {5, 11, 1}, s1}}}, {{{s0, {5, 11, 6}, s1}}}, true);
  baseTest({{{s0, {5, 11, 1}, s1}}}, {{{s0, {5, 11, 5}, s1}}}, false);
}

} // namespace

int main() {
  testMergeA0();
  testMergeA1();
  testMergeB0();
  testMergeC0();

  Sett a{{{{2, 10, 4}}}};
  Sett b{{{{3, 9, 6}, {1, 1, 0}}}};
  auto foo = Sett::transfer(a, b);

  return 0;
}
