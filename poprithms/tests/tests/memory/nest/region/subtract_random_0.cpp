// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <random>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/unisort.hpp>
#include <testutil/memory/nest/randomregion.hpp>

namespace {

using namespace poprithms::memory::nest;

void test() {

  std::mt19937 gen(11011);

  for (uint64_t ti = 0; ti < 1024; ++ti) {
    std::cout << ti << ' ';
    if (ti % 26 == 25) {
      std::cout << std::endl;
    }

    const auto reg0 = getRandomRegion(
        {5, 6, 7, 8}, /* seed */ 100 + ti, /* max Sett depth */ 3);

    const auto reg1 = getRandomRegion(
        {5, 6, 7, 8}, /* seed */ 10000 + ti, /* max Sett depth */ 3);

    const auto A = reg0.intersect(reg1);
    const auto B = reg1.subtract(reg0);
    const auto C = reg0.subtract(reg1);

    if (2 * A.totalElms() + B.totalElms() + C.totalElms() !=
        reg0.totalElms() + reg1.totalElms()) {
      std::ostringstream oss;
      oss << "Failure in random region subtraction test. "
          << " This with \n  reg0 = " << reg0 << "  reg1 = " << reg1 << ".";
      throw poprithms::test::error(oss.str());
    }
  }
}

} // namespace

int main() {
  test();
  return 0;
}
