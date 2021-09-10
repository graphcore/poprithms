// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <random>
#include <sstream>

#include <testutil/memory/nest/randomsett.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/unisort.hpp>

namespace {

using namespace poprithms::memory::nest;

void test(int max0, uint64_t nTests, bool exact, bool doCount) {

  std::mt19937 gen(11011);

  const bool shorten = true;

  std::cout << "test number with nTests = " << nTests << " exact = " << exact
            << " doCount = " << doCount << "\n-------------\n";
  for (uint64_t ti = 0; ti < nTests; ++ti) {
    std::cout << ti << ' ';
    if (ti % 16 == 15) {
      std::cout << std::endl;
    }

    const auto depth0 = 0 + gen() % 3;
    const auto depth1 = 0 + gen() % 3;

    const bool canonicalize0 = gen() % 1;
    const bool canonicalize1 = gen() % 1;

    const auto p0 = poprithms::memory::nest::getRandom(
        shorten, depth0, canonicalize0, ti + 100, max0);

    const auto p1 = poprithms::memory::nest::getRandom(
        shorten, depth1, canonicalize1, ti + 1000, max0);

    const auto A = p0.intersect(p1);
    const auto B = p1.subtract(p0);
    const auto C = p0.subtract(p1);

    if (exact) {
      for (const auto &a : A.get()) {
        a.confirmDisjoint(B.get());
        a.confirmDisjoint(C.get());
      }

      for (const auto &b : B.get()) {
        b.confirmDisjoint(C.get());
      }
    }
    if (doCount) {
      const auto lcm = p0.smallestCommonMultiple(p1);
      if (2 * A.totalOns(lcm) + B.totalOns(lcm) + C.totalOns(lcm) !=
          p0.n(lcm) + p1.n(lcm)) {
        throw poprithms::test::error(
            "Unexpected counts in random subtract test");
      }
    }
  }
}

} // namespace

int main() {
  test(13, 128, true, false);
  test(50, 512, false, true);
  return 0;
}
