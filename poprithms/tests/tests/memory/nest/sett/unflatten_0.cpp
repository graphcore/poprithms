// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <sstream>

#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {

poprithms::logging::Logger logger("unflattest");
using namespace poprithms::memory::nest;

void appendCounts(std::ostringstream &oss,
                  const std::vector<int> &counts,
                  uint64_t p0) {
  for (uint64_t i = 0; i < counts.size() / p0; ++i) {
    oss << '\n' << "            ";
    for (uint64_t j = 0; j < p0; ++j) {
      oss << counts[i * p0 + j] << " ";
    }
  }
}

void test0() {

  // ..x..
  // ..x..
  // .....
  // ..x..
  // ..x..
  // .....
  logger.info("test #0");
  Sett sett0({{10, 5, 0}, {1, 4, 2}});
  auto unflattened = sett0.unflatten(5);
  if (unflattened.size() != 1) {
    throw error("Expected just 1 setting in test0");
  }
  const auto p0 = std::get<0>(unflattened[0]);
  const auto p1 = std::get<1>(unflattened[0]);
  p0.confirmEquivalent({{{2, 1, 0}}});
  p1.confirmEquivalent({{{1, 4, 2}}});
}

void test1() {

  // x...xx...xx...x
  // x...xx...xx...x
  // ...............
  // x...xx...xx...x
  // ................
  // x...xx...xx...x
  // x...xx...xx...x
  // ...............
  // x...xx...xx...x
  logger.info("test #1");
  Sett sett0({{75, 0, 15}, {15, 15, 0}, {2, 3, -1}});
  auto unflattened = sett0.unflatten(15);
  if (unflattened.size() != 1) {
    throw error("Expected just 1 setting in test0");
  }
  const auto p0 = std::get<0>(unflattened[0]);
  const auto p1 = std::get<1>(unflattened[0]);
  p0.confirmEquivalent({{{5, 0, 1}, {1, 1, 0}}});
  p1.confirmEquivalent({{{2, 3, -1}}});
}

void test2() {

  logger.info("test #2");

  // ..........
  // .xxxxxxxx.
  // .xxxxxxxx.
  // .xxxxxxxx.
  // .xxxxxxxx.
  // .xxxxxxxx.
  // .xxxxxxxx.
  // .xxxxxxxx.
  // .xxxxxxxx.
  // ..........
  //
  // becomes
  //
  // ...........xxxxxxxx.
  // .xxxxxxxx..xxxxxxxx.
  // .xxxxxxxx..xxxxxxxx.
  // .xxxxxxxx..xxxxxxxx.
  // .xxxxxxxx...........
  //

  const int64_t p0 = 20;
  Sett sett0({{80, 20, 10}, {8, 2, 1}});
  auto uf = sett0.unflatten(p0);
  if (p0 == 20 && uf.size() > 3) {
    // ((1,4,0))  ((10,10,10)(8,2,1))
    // ((3,2,1))  ((8,2,1))
    // ((1,4,4))  ((10,10,0)(8,2,1))
    std::ostringstream oss;
    oss << "Unflattened can be expressed with just 3, "
        << "but uf.size() = " << uf.size() << ".";
    for (const auto &x : uf) {
      oss << "\n      " << std::get<0>(x) << "  " << std::get<1>(x);
    }

    throw error(oss.str());
  }

  auto counts = Sett::getRepeatingOnCount(Sett::scaledConcat(uf, p0));
  std::ostringstream oss;
  appendCounts(oss, counts, p0);
  logger.trace(oss.str());
  const auto upscaled = Sett::scaledConcat(uf, p0);
  Sett::confirmDisjoint(upscaled);
  sett0.confirmEquivalent(DisjointSetts(upscaled));
}

void test3() {
  for (int64_t p0 : {2, 5, 7, 15, 40}) {
    Sett sett0({{80, 20, 10}, {8, 2, 1}});
    auto uf = sett0.unflatten(p0);
    std::ostringstream oss;
    auto counts = Sett::getRepeatingOnCount(Sett::scaledConcat(uf, p0));
    appendCounts(oss, counts, p0);
    logger.trace(oss.str());
    const auto upscaled = Sett::scaledConcat(uf, p0);
    Sett::confirmDisjoint(upscaled);
    sett0.confirmEquivalent(DisjointSetts(upscaled));
  }
}

void test4() {
  Sett foo{{{11664, 0, 0}, {5189, 3285, 2768}, {1508, 1680, 2810}}};
  auto unflat = foo.unflatten(972);

  //   Sett foo{{{{1000,1000,0}}}};
  //   auto unflat = foo.unflatten(1011);
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
}
