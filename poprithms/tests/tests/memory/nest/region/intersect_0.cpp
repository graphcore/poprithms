// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

void assertIntersection(const Region &a,
                        const Region &b,
                        const DisjointRegions &inter) {

  auto foo = a.intersect(b);

  if (!Region::equivalent(inter, foo)) {
    std::ostringstream oss;
    oss << "in test for Region::equivalent, detected failure. "
        << "Expected the intersection of " << a << " and " << b << " to be "
        << inter << ", but it is " << foo << ".";
    throw error(oss.str());
  }

  if (inter.size() < foo.size()) {
    std::ostringstream oss;
    oss << "Got the correct intersection between " << a << " and " << b
        << " in this test for Region intersection, "
        << " but the expected solution is more compact. ";
    throw error(oss.str());
  }
}

void test0() {
  // ..xxxxx...
  // ....xxxxx.
  Region a({10, 10}, {{{{5, 5, 2}}}, {{{5, 5, 2}}}});
  Region b({10, 10}, {{{{5, 5, 4}}}, {{{5, 5, 4}}}});
  Region expectedIntersection({10, 10}, {{{{3, 7, 4}}}, {{{3, 7, 4}}}});
  assertIntersection(a, b, expectedIntersection);
}

void test1() {
  // .....
  // .xx..
  // .xx..
  // .....
  Region a({4, 5}, {{{{2, 2, 1}}}, {{{3, 2, 1}}}});

  // xxxxx
  // .....
  // .....
  // xxxxx
  Region b({4, 5}, {{{{2, 2, -1}}}, {{{1, 0, 0}}}});
  Region expectedIntersection({4, 5}, {{{{0, 1, 0}}}, {{{0, 1, 0}}}});
  assertIntersection(a, b, expectedIntersection);
}

} // namespace

int main() {

  using namespace poprithms::memory::nest;

  test0();
  test1();

  return 0;
}
