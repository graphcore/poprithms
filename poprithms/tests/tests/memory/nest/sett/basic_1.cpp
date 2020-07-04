// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {

using namespace poprithms::memory::nest;

void assertSCM(const Sett &a, const Sett &b, int64_t expected) {

  auto computed = a.smallestCommonMultiple(b);
  if (computed != expected) {
    std::ostringstream oss;
    oss << "Expected " << a << ".smallestCommonMultiple(" << b << ") to be "
        << expected << ", not " << computed << std::endl;
    throw error(oss.str());
  }
}

void assertSCM_v(const std::vector<Sett> &sett0s, int64_t expected) {
  auto computed = Sett::smallestCommonMultiple_v(sett0s);
  if (computed != expected) {
    std::ostringstream oss;
    oss << "Expected "
        << "Sett::smallestCommonMultiple(" << sett0s << ") to be " << expected
        << ", not " << computed << std::endl;
    throw error(oss.str());
  }
}

void testSmallestCommonMultiple() {
  // 14 + 2 = 16
  // 6 + 1 = 7
  assertSCM({{{14, 2, 5}}}, {{{6, 1, 22}}}, 16 * 7);

  // canonicalization will reduce the second Sett to the stripeless
  // Sett.
  assertSCM({{{14, 2, 5}}}, {{{7, 0, 22}}}, 16);

  // nested Stripes have no effect on SCM.
  assertSCM({{{14, 2, 5}}}, {{{6, 1, 22}, {2, 1, 5}}}, 16 * 7);

  // 8 is a factor of 16, so SCM here is 16.
  assertSCM({{{14, 2, 5}}}, {{{7, 1, 22}, {2, 1, 5}}}, 16);

  // 3, 5, 7, 9, 11
  assertSCM_v({{{{1, 2, 3}}},
               {{{2, 3, 4}}},
               {{{3, 4, 5}}},
               {{{4, 5, 6}}},
               {{{5, 6, 7}}}},
              3 * 5 * 7 * 3 * 11);
}

void testEquivalence0() {
  // 01234567890123456789012345678901234567890
  // 111...11..1.111...11..1.111...11..1.111...11..1.
  // xxxxxxxxxxx                   xxxxxxxxx
  // xxxxxxxx  x                   xx..xxxxx
  // xxx   xx                      xx  x.xxx
  // a is the left interpretation, b is the right interpretation.
  Sett a{{{{11, 1, 0}, {8, 2, 0}, {3, 3, 0}}}};
  Sett b{{{{9, 3, 6}, {7, 2, 4}, {100, 1, 2}}}};
  // A different Sett:
  Sett c{{{{9, 3, 6}, {7, 2, 4}, {100, 2, 2}}}};
  a.confirmEquivalent(b);
  if (!a.equivalent(b)) {
    throw error(
        "expected equivalence in testEquivalence0, and moreover expected the "
        "difference to be caught in conirmEquivalent above");
  }
  if (a.equivalent(c)) {
    throw error("Expected non-equivalence in testEquivalence0");
  }
}

void testAlwaysOff() {
  Sett a{{{0, 1, 0}}};
  Sett b{{{1, 2, 0}, {1, 1, 1}}};
  Sett c{{{1, 2, 0}, {1, 1, 0}}};

  if (a.alwaysOff() && b.alwaysOff() && !c.alwaysOff()) {
  } else {
    throw error("Failure in testAlwaysOff");
  }
}

} // namespace

int main() {

  testEquivalence0();
  testAlwaysOff();
  testSmallestCommonMultiple();

  return 0;
}
