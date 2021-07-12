// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {
using namespace poprithms::memory::nest;
void assertNotEquiv(const Sett &a, const Sett &b) {
  if (a.equivalent(b)) {
    std::ostringstream oss;
    oss << "Expected(" << a << ".equivalent(" << b << ") to be false";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {
  using namespace poprithms::memory::nest;

  assertNotEquiv({{{10, 7, 2}}}, {{{10, 7, 3}}});
  assertNotEquiv({{{100, 1, 0}, {1, 1, 0}}}, {{{1, 1, 0}}});
  assertNotEquiv({{{100, 1, 0}, {1, 1, 0}}}, {{{11, 1, 0}, {1, 1, 0}}});
  assertNotEquiv({{{100000, 1, 0}, {1, 1, 0}}},
                 {{{100000, 2, 0}, {1, 1, 0}}});
  assertNotEquiv({{{100000, 3, 0}, {1, 1, 0}}},
                 {{{100000, 3, 2}, {1, 1, 0}}});
  assertNotEquiv({{{100000, 1, 0}, {1, 1, 0}}},
                 {{{100000, 1, 2}, {1, 1, 0}}});
  assertNotEquiv({{{100000, 0, 0}, {1, 1, 0}}},
                 {{{100000, 1, 1}, {1, 1, 1}}});
  assertNotEquiv({{{100000, 0, 0}, {2, 1, 0}}},
                 {{{100001, 0, 0}, {2, 1, 0}}});

  Sett p{{{10, 2, 1}, {3, 2, 2}}};
  p.confirmEquivalent(p);
}

void test1() {
  auto b = Sett::createAlwaysOn();
  if (b.hasStripes()) {
    throw poprithms::test::error("No, b does not have stripes.");
  }

  Sett c({{1, 2, 3}});
  if (!c.hasStripes()) {
    throw poprithms::test::error("Yes, c does have stripes");
  }
}

void testEquiv0() {
  Sett x0{{{1, 1, 0}}};
  Sett x1{{{4, 2, 1}, {1, 1, 1}}};
  Sett x2{{{1, 5, 0}}};
  if (!x0.equivalent(DisjointSetts({x1, x2})) || x0.equivalent(x1)) {
    throw poprithms::test::error("Failed in testEquiv0");
  }
  x0.confirmEquivalent(DisjointSetts({x1, x2}));
}

void testContains0() {
  Sett sett0{{{10, 10, 0}, {2, 2, 0}}};
  Sett p2d{{{10, 30, 0}, {2, 2, 0}}};
  Sett p3d{{{10, 30, 0}, {2, 3, 0}}};
  if (!sett0.contains(p2d)) {
    throw poprithms::test::error("Failed in testContains0 (A)");
  }
  if (sett0.contains(p3d)) {
    throw poprithms::test::error("Failed in testContains0 (B)");
  }
}

void testContainedInDisjoint0() {

  Sett sett0{{{10, 10, 0}, {1, 1, 0}}};
  Sett p2d{{{10, 30, 0}, {1, 1, 0}}};
  Sett p3d{{{15, 25, 20}, {1, 1, 0}}};
  if (!sett0.containedIn(DisjointSetts({p2d, p3d}))) {
    throw poprithms::test::error("Failed in testContainedInDisjoint0");
  }
}

void testHasStripes() {
  Sett p0{{}};
  Sett p1{{{4, 2, 3}}};
  if (p0.hasStripes() || !p1.hasStripes()) {
    throw poprithms::test::error("failed in test of has stripes");
  }
}

void assertN(const Sett &s, int64_t a, int64_t b, int64_t n) {
  if (s.n(a, b) != n) {
    std::ostringstream oss;
    oss << "Expected " << s << " to contain " << n << " ons (\'1\'s) "
        << "in the range [" << a << ", " << b << ')';
    throw poprithms::test::error(oss.str());
  }
}

void testN() {
  assertN({{}}, 14, 15, 1);
  assertN({{}}, 14, 14, 0);
  assertN({{}}, 14, 24, 10);
  assertN({{{1, 1, 0}}}, 10, 11, 1);
  assertN({{{1, 1, 0}}}, 11, 12, 0);
  assertN({{{5, 0, 4}}}, 11, 12, 1);
  assertN({{{5, 0, 4}}}, 10, 13, 3);
}

} // namespace

int main() {
  test0();
  test1();
  testEquiv0();
  testContains0();
  testContainedInDisjoint0();
  testN();
  testHasStripes();
  return 0;
}
