// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <random>

#include <poprithms/memory/nest/error.hpp>
#include <testutil/memory/nest/randomsett.hpp>

int main() {
  // In this test, we assert that getOns(start, end) agrees with find(x)
  // for x in [start, end).
  // In particular, we assert that find(x) is the smallest value in
  // getOns(start, end) greater than or equal to x.

  using namespace poprithms::memory::nest;

  bool shorten = true;
  int max0     = 200;

  for (uint64_t ti = 0; ti < 2048; ++ti) {
    if (ti % 16 == 0) {
      std::cout << std::endl;
    }
    std::cout << ' ' << ti;
    std::mt19937 gen(ti + 10101);
    auto depth0        = 0 + gen() % 4;
    bool canonicalize0 = true;
    auto sett          = poprithms::memory::nest::getRandom(
        shorten, depth0, canonicalize0, ti + 100, max0);

    auto ons = sett.getOns(0, max0);

    if (ons.size() != 0) {
      for (uint64_t i0 = 0; i0 < ons.size() - 1; ++i0) {
        int64_t x0 = ons[i0];
        int64_t x1 = ons[i0 + 1];
        for (auto x = x0 + 1; x <= x1; ++x) {
          if (sett.find(x) != x1) {
            std::ostringstream oss;
            oss << "Failure in test of Sett:find. For sett=" << sett << ". ";
            oss << "Expected " << sett << ".find(" << x << ") to be " << x1
                << ", not " << sett.find(x);
            throw error(oss.str());
          }
        }
      }
    } else {
      std::cout << '-';
    }
  }

  return 0;
}
