// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <numeric>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

void testMultiOutSoln(const Sett &scaffold,
                      const Sett &ink,
                      const std::vector<Sett> &expected,
                      uint64_t maxAllowedParts,
                      std::string name) {

  auto filled = scaffold.fillWith(ink);

  std::ostringstream oss;
  oss << "\nIn testMultiOutSoln, sub-test " << name << ", with";
  oss << "\n   Scaffold=" << scaffold << "\n   Ink=" << ink
      << "\n   Expected={";
  for (const auto &x : expected) {
    oss << ' ' << x << ' ';
  }
  oss << '}';
  oss << "\n   Observed={";
  for (const auto &x : filled) {
    oss << ' ' << x << ' ';
  }
  oss << '}';
  std::cout << oss.str() << '.' << std::endl;

  auto scm0 = Sett::smallestCommonMultiple_v(expected);
  auto scm1 = Sett::smallestCommonMultiple_v(filled.get());
  auto scm  = smallestCommonMultiple_i64(scm0, scm1);

  auto getn = [scm](const DisjointSetts &xs) {
    return std::accumulate(
        xs.cbegin(), xs.cend(), int64_t(0), [scm](int64_t n0, const Sett &x) {
          return n0 + x.n(0, scm);
        });
  };

  Sett::confirmDisjoint(expected);
  if (getn(DisjointSetts(expected)) != getn(filled)) {
    throw error("Different counts in [0, scm)");
  }

  for (const auto &x : expected) {
    if (!x.containedIn(filled)) {
      throw error("Not identical elements");
    }
  }

  if (filled.size() > maxAllowedParts) {
    throw error("Correct, but expected at most " +
                std::to_string(maxAllowedParts) + "  in output of fillWith");
  }
}

void testSingletonSoln(const Sett &scaffold,
                       const Sett &ink,
                       const Sett &expected,
                       int tid) {

  {
    std::ostringstream oss;
    oss << "\nIn testSingletonSoln, sub-test " << tid << ", with";
    oss << "\n   Scaffold=" << scaffold << "\n   Ink=" << ink
        << "\n   Expected=" << expected << ". Getting filled... ";
    std::cout << oss.str() << std::endl;
  }

  auto filled = scaffold.fillWith(ink);

  {

    std::ostringstream oss;
    oss << "\n  Observed={";
    for (const auto &x : filled) {
      oss << ' ' << x << ' ';
    }
    oss << '}';
    std::cout << oss.str() << '.' << std::endl;
  }

  if (filled.size() != 1) {
    throw error("expected just 1 in output of fillWith");
  }

  if (!expected.equivalent(filled[0])) {
    throw error("not as expected");
  }
}
} // namespace

void singletonTests() {
  using namespace poprithms::memory::nest;

  testSingletonSoln({{{4, 4, 0}}}, {{{2, 2, 0}}}, {{{2, 6, 0}}}, 0);
  // full scaffold
  testSingletonSoln({{}}, {{{2, 1, 0}}}, {{{2, 1, 0}}}, 1);
  // full ink
  testSingletonSoln({{{2, 3, 4}}}, {{}}, {{{2, 3, 4}}}, 2);

  // xx....xxxx....xxxx....xxxx....
  // xx    ..xx    ..xx    ..xx
  testSingletonSoln({{{4, 4, 6}}}, {{{2, 2, 0}}}, {{{2, 6, 0}}}, 3);

  // x.x.x.x.x.x.x.x.x.x.x.x.x.x.x.x   scaffold
  // x x x x . . . . x x x x . . . .   ink
  // x.x.x.x.........x.x.x.x........   expected
  testSingletonSoln(
      {{{1, 1, 0}}}, {{{4, 4, 0}}}, {{{8, 8, 0}, {1, 1, 0}}}, 4);
}

void multiOutTests() {

  // x.xxx.xxx.xxx.xxx. scaff
  // x .x. x.x .x. x.x  ink
  // x..x..x.x..x..x.x. result, which is made up of 2 spawns:
  //
  // x.....xxx.....xxx. from (3,5,6) spawn,
  // x     x.x     x.x         == (3,5,6)(1,1,0)
  //
  // ..xxx.....xxx..... from (3,5,2) spawn.
  //   .x.     .x.             == (1,7,3).
  //
  //
  // 14:    Observed={ ((1,7,3))  ((3,5,6)(1,1,0)) }.
  // ((3,5,6)(3,0,2)(1,1,0)
  //

  Sett p0{{{1, 7, 3}}};
  Sett p1{{{3, 5, 6}, {1, 1, 0}}};
  testMultiOutSoln({{{3, 1, 2}}}, {{{1, 1, 0}}}, {p0, p1}, 2, "multi0");

  // x.xxx.xxx.xxx.xxx.xxx.xxx.xxx.xxx.xxx.xxx.xxx.xxx scaffold
  // x ... ..x .x. x.x ... ..x .x. x.x ... ..x .x. x.x ink
  // x.......x..x..x.x.......x..x..x.x.......x..x..x.x soln.
  //...........x...............x...............x..... spawn 0 (1, 15, 11)
  // x ......x.....x.x.......x.....x.x.......x.....x.x spawn 1
  //                                               (10,6,8)(3,5,6)(1,1,0)
  // 0123456789012345678901234567890123456789
  // 0         10        20        30
  //
  //(10,6,8)(3,5,6)(1,1,0)
  // xx......xxxxxxxxxx......xxxxxxxxxx......
  // x.      x.....xxx.      x.....xxx.
  // x       x     x.x       .x    x.x
  p0 = {{{1, 15, 11}}};
  p1 = {{{10, 6, 8}, {3, 5, 6}, {1, 1, 0}}};
  testMultiOutSoln(
      {{{3, 1, 2}}}, {{{7, 5, 6}, {1, 1, 0}}}, {p0, p1}, 2, "multi1");

  // ((3,12,2)(1,1,0)) ((2,13,10)) ((7,8,0)(1,5,0))

  // ........xxxxxxxxxx.........xxxxxxxxxx.........(10, 9, 8)
  //         xx.xxx.xxx                            (3, 1, 3)
  //
  //         x. x.x .x.
  // ........x..x.x..x..........x..x.x..x.
  // 01234567890123456789012345678901234567890
  //
  // ...........x.x................ (3, 16, 11)(1,1,0)
  // ........x.......x..........    (9,10,8)(1,7,0)
  // ((3,16,11)(1,1,0)) ((9,10,8)(1,7,0))
  //
  testMultiOutSoln({{{10, 9, 8}, {3, 1, 3}}},
                   {{{1, 1, 0}}},
                   {{{{3, 16, 11}, {1, 1, 0}}}, {{{9, 10, 8}, {1, 7, 0}}}},
                   2,
                   "multi2");
}

void fillEmptyTest() {

  // Always off:
  Sett scaffold{{{0, 100, 0}}};

  // Sometimes on:
  Sett ink{{{3, 6, 2}}};

  const auto filled = scaffold.fillWith(ink);
  if (!filled.empty()) {
    throw error("Filling an empty Sett results in an empty Sett");
  }
}

int main() {
  singletonTests();
  multiOutTests();
  fillEmptyTest();
  return 0;
}
