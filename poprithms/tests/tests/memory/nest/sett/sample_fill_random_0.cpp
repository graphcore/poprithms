// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poprithms/logging/logging.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/logging.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>
#include <testutil/memory/nest/randomsett.hpp>

namespace {

poprithms::logging::Logger logger("testLog");

using namespace poprithms::memory::nest;

void confirmEquivalent(const Sett &scaff, const Sett &ink) {

  logger.setLevelOff();
  bool doLog = false;
  if (doLog) {
    logger.setLevelInfo();
    std::cout << "\n" << std::endl;
  }

  poprithms::logging::enableDeltaTime(true);
  std::ostringstream oss;
  oss << "scaff=" << scaff << ", ink=" << ink << ". Entering fill.";
  logger.info(oss.str());
  auto filled = Sett::fill(scaff, ink);
  std::vector<Sett> allSampled;
  logger.info("Entering sample");
  for (const auto &f : filled) {
    auto sampled = Sett::sample(f, scaff);
    allSampled.insert(allSampled.end(), sampled.cbegin(), sampled.cend());
  }
  logger.info("Entering confirmDisjoint for " +
              std::to_string(allSampled.size()) + " Setts");
  Sett::confirmDisjoint(allSampled);
  logger.info("Entering confirmEquivalent");
  ink.confirmEquivalent(DisjointSetts(allSampled));
}
} // namespace

void test0() {
  Sett scaff{{{7, 4, 3}, {1, 1, 1}}};
  Sett ink{{{3, 1, 0}}};
  confirmEquivalent(scaff, ink);
}

// Takes a few seconds still:
// void test1() {
//   Sett scaff{{{{2260, 2436, 1353}, {2294, 1089, 1803}}}};
//   Sett ink{{{{3473, 1368, 3711}}}};
//   confirmEquivalent(scaff, ink);
// }

void testRandom() {

  // Setts will be of depth [lDepth, uDepth)
  uint64_t lDepth = 0;
  uint64_t uDepth = 4;

  // The maximum "on" for the first Stripe in Setts.
  int gmaxon = 32;

  int seed = 1000;
  std::mt19937 gen(seed);

  // run to 150,000 on 27 May 2020: record this. Not fast enough to
  // run regularly for this number of iterations.
  for (int i = 0; i < 512; ++i) {

    std::cout << i << " " << std::flush;
    if (i % 16 == 0) {
      std::cout << std::endl;
    }

    bool shorten0 = gen() % 2;
    bool shorten1 = gen() % 2;

    int64_t depth0 = lDepth + gen() % (uDepth - lDepth);
    int64_t depth1 = lDepth + gen() % (uDepth - lDepth);

    bool canon0 = gen() % 2;
    bool canon1 = gen() % 2;

    int seed0 = gen() % 1000000;
    int seed1 = gen() % 1000000;

    Sett ink = poprithms::memory::nest::getRandom(
        shorten0, depth0, canon0, seed0, gmaxon);

    Sett scaffold = poprithms::memory::nest::getRandom(
        shorten1, depth1, canon1, seed1, gmaxon);

    if (!scaffold.alwaysOff()) {
      confirmEquivalent(scaffold, ink);
    }
  }
}

int main() {
  test0();
  testRandom();
  return 0;
}
