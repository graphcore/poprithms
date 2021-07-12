// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/printiter.hpp>
#include <testutil/memory/nest/randomregion.hpp>

// to test:
// settSample
// settFillInto
// settFillWith
//
// facts to assert:
//
// for a,b of same shape:
// 1) a.contains(a.settSample(b).fillInto(b))
// 2) b.settSample(x).isAlwaysOn() for x in a.intersect(b).
//
// for c of shape b.nelms():
// 3) b.contains(b.fillWith(c))
// 4) b.fillWith(c).equivalent(c.fillInto(b))

namespace {

void test() {

  poprithms::logging::Logger logger("testLogger");
  poprithms::logging::enableDeltaTime(true);

  bool doLog = true;
  if (doLog) {
    logger.setLevelInfo();
  }

  using namespace poprithms::memory::nest;
  const auto maxSettDepth = 3;
  uint64_t seed           = 1000U;
  for (uint32_t i = 0; i < 200; ++i) {

    {
      std::ostringstream oss;
      oss << "i=" << i;
      logger.info(oss.str());
    }

    seed += 10;
    const auto shapes = getShapes(seed + 1, 3, 3, 4, 12);
    const auto shape0 = std::get<0>(shapes);
    const auto a      = getRandomRegion(shape0, seed + 2, maxSettDepth);
    const auto b      = getRandomRegion(shape0, seed + 3, maxSettDepth);

    const auto sampled0 = a.settSample(b);
    const auto inter    = a.intersect(b);

    logger.info("test 1");
    for (const auto &x : sampled0.get()) {
      auto foo = x.settFillInto(b);
      for (const auto &reg : foo.get()) {
        if (!a.contains(reg)) {
          throw poprithms::test::error("failed test 1");
        }
      }
    }

    logger.info("test 2");
    for (const auto &x : inter.get()) {
      const auto sampled = a.settSample(x);
      if (!Region::equivalent(sampled,
                              {Region::createFull(sampled.shape())})) {
        std::ostringstream oss;
        oss << "\na = " << a << '\n';
        oss << "b = " << b << '\n';
        oss << "part of inter (x) = " << x << '\n';
        oss << "a.settSample(x) = " << sampled << '\n';
        throw poprithms::test::error(oss.str());
      }
    }

    const auto nelmsb = b.nelms();
    const auto c      = getRandomRegion(nelmsb, seed + 4, maxSettDepth);
    const auto filled = b.settFillWith(c);

    logger.info("test 3");
    for (const auto &f : filled.get()) {
      if (!b.contains(f)) {
        throw poprithms::test::error("error test 3 ");
      }
    }

    logger.info("test 4");
    std::ostringstream oss;
    oss << "\nfilled = " << filled;
    oss << "\nc filled into = " << c.settFillInto(b);
    logger.info(oss.str());

    if (!Region::equivalent(filled, c.settFillInto(b))) {
      throw poprithms::test::error("error test 4");
    }
  }
}

} // namespace

int main() {
  test();
  return 0;
}
