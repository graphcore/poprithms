// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <testutil/memory/nest/randomsett.hpp>

// Random tests that Sett canonicalization is valid.
int main() {

  using namespace poprithms::memory::nest;

  // Maximum "on" of first Stripe
  int64_t max0 = 100;

  // How many Stripes should the Sett contain?
  for (int64_t depth : {1, 2, 3, 4}) {

    // What is the periodicity on which the Sett should be split?
    for (int64_t period : {3, 10}) {

      // How many tests with this (shorten, depth) setting?
      for (uint64_t nTests = 0; nTests < 128; ++nTests) {

        // non-canonicalized:
        Sett sett =
            poprithms::memory::nest::getRandom(true,       // shorten
                                               depth,      // recursive depth
                                               true,       // canonicalized
                                               1 + nTests, // seed
                                               max0);

        const auto unflattened = sett.unflatten(period);
        const auto reflattened = Sett::scaledConcat(unflattened, period);
        Sett::confirmDisjoint(reflattened);
        sett.confirmEquivalent(DisjointSetts(reflattened));
      }
    }
  }
}
