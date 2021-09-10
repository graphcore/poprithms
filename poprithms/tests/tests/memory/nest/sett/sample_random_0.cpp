// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <testutil/memory/nest/randomsett.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

int main() {

  using namespace poprithms::memory::nest;

  for (uint32_t seed = 0; seed < 512; ++seed) {

    const Sett x =
        poprithms::memory::nest::getRandom(true, 3, true, seed, 32);

    Sett where =
        poprithms::memory::nest::getRandom(true, 3, true, seed + 1, 32);

    if (where.alwaysOff()) {
      continue;
    }

    // 11..1.11.1..111.11.1..1.11.1.11.11.1...111111   x
    // ...11.11..11.11...1.11.11.11.1...11....1..1..   where
    //    .1 11  .. 11   . .. .1 .1 1   1.    1  1
    // .111..11....1.111.11                            sampled
    // 1,2,3,6,7 etc                                   sampledIndices

    auto sampled = x.sampleAt(where);

    auto scm = Sett::smallestCommonMultiple_v(sampled.get());
    std::vector<int64_t> sampledIndices;
    for (const auto &s : sampled) {
      auto inds = s.getOns(0, scm);
      sampledIndices.insert(sampledIndices.end(), inds.cbegin(), inds.cend());
    }
    std::sort(sampledIndices.begin(), sampledIndices.end());

    auto topIndex = where.getOn(scm);
    auto whereOn  = where.getOns(0, topIndex);
    auto xOn      = x.getOns(0, topIndex);
    std::vector<bool> xAsBit(topIndex, false);
    for (auto i : xOn) {
      xAsBit[i] = true;
    }

    std::vector<int64_t> alternate;
    for (uint64_t j = 0; j < whereOn.size(); ++j) {
      if (xAsBit[whereOn[j]]) {
        alternate.push_back(static_cast<int64_t>(j));
      }
    }

    if (alternate != sampledIndices) {
      std::ostringstream oss;
      oss << "Failed with x = " << x << " and where = " << where
          << ". sampled with baseline : ";
      for (auto v : alternate) {
        oss << v << " ";
      }
      oss << "\n sampled directly : ";
      for (auto c : sampledIndices) {
        oss << c << " ";
      }
      oss.str();
    }
  }

  return 0;
}
