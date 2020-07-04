// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <numeric>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

void testDisjoint(bool expected, const std::vector<Sett> &b) {
  auto computed = Sett::disjoint(b);
  if (computed != expected) {
    std::ostringstream oss;
    oss << "In testDisjoint, testing disjointedness of Setts in " << b
        << ", expected = " << expected << ". But computed = " << computed
        << '.';
    throw error(oss.str());
  }
}

void test0() {

  // largest common factor of periods is 6
  int64_t b = 6000000;
  testDisjoint(
      1, {{{{b - 17, 17, 0}, {1, 5, 0}}}, {{{1, 5, 1}}}, {{{1, 5, 2}}}});

  std::cout << "done" << std::endl;

  // forms support if all integers
  testDisjoint(1, {{{{3, 7, 0}}}, {{{3, 7, 3}}}, {{{4, 6, 6}}}});
  //                                                               | mutate
  //                                                               |
  testDisjoint(0, {{{{3, 7, 0}}}, {{{3, 7, 3}}}, {{{4, 6, 5}}}});
  //                                                               | mutate
  //                                                               |
  testDisjoint(0, {{{{3, 7, 0}}}, {{{3, 7, 3}}}, {{{4, 6, 7}}}});

  testDisjoint(1,
               {{{{10, 10, 0}}},
                {{{10, 10, 10}, {3, 7, 0}}},
                {{{10, 10, 10}, {3, 7, 3}}},
                {{{10, 30, 10}, {4, 6, 6}}},
                {{{10, 30, 30}, {3, 7, 7}}}});

  auto base = 10000;

  // largest common factor of periods is 2. One has on at an even index,
  // one at an odd index: no intersect
  testDisjoint(1, {{{{1, base + 1, 0}}}, {{{1, base + 3, 1}}}});
  testDisjoint(1, {{{{1, base + 1, 10}}}, {{{1, base + 3, 1}}}});
  testDisjoint(1, {{{{1, base + 1, 10}}}, {{{1, base + 3, 101}}}});

  // if they both have ons at even (or both at odd) then there is an
  // intersection
  testDisjoint(0, {{{{1, base + 1, 3}}}, {{{1, base + 3, 1}}}});
  testDisjoint(0, {{{{1, base + 1, 6}}}, {{{1, base + 3, 8}}}});
}
} // namespace

int main() {

  test0();
  return 0;
}
