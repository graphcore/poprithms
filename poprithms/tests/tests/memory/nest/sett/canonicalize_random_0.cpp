// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <testutil/memory/nest/randomsett.hpp>

// Random tests that Sett canonicalization is valid.
int main() {

  using namespace poprithms::memory::nest;

  int64_t max0 = 20;

  // Should subsequent Stripes in the Sett be strictly shorter?
  for (bool shorten : {false, true}) {

    // How many Stripes should the Sett contain?
    for (int64_t depth : {2, 3, 4}) {

      // How many tests with this (shorten, depth) setting?
      for (uint64_t nTests = 0; nTests < 2048; ++nTests) {

        // non-canonicalized:
        Sett nCan = poprithms::memory::nest::getRandom(
            shorten, depth, false, 1 + nTests, max0);

        // canonicalized:
        Sett can = poprithms::memory::nest::getRandom(
            shorten, depth, true, 1 + nTests, max0);

        // get scm range, and check over slightly larger range:
        int64_t range = nCan.atDepth(0).period();
        if (can.hasStripes()) {
          range = smallestCommonMultiple_i64(nCan.atDepth(0).period(),
                                             can.atDepth(0).period());
        }
        if (nCan.getOns(7, 11 + 2 * range) != can.getOns(7, 11 + 2 * range)) {
          std::ostringstream oss;
          oss << "Failure in comparison of getOns(0, " << range
              << ") between canonicalized Sett, \n   " << can
              << " and non-canonicalized Sett, \n   " << nCan << ".";
          throw error(oss.str());
        }
      }
    }
  }
}
