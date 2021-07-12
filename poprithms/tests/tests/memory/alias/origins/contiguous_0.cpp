// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/origins.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {
  Origins oris({10, 20});
  oris.insert(AllocId(1), DisjointRegions::createFull({20, 1}));
  if (oris.isRowMajorSetContiguous()) {
    throw poprithms::test::error(
        "Must contain self-aliases (20 < 20*10), and so cannot be "
        "row major set contiguous");
  }

  Origins oris2({100, 2});
  oris2.insert(AllocId(17), DisjointRegions::createFull({25, 8}));
  if (!oris2.isRowMajorSetContiguous()) {
    throw poprithms::test::error(
        "oris2 should be row major set contiguous (100*2 = 25*8)");
  }

  Origins oris3({100, 2});
  oris3.insert(AllocId(6), Region::fromStripe({1, 1, 200}, 2, {1, 1, 0}));
  oris3.insert(AllocId(6), Region::fromStripe({1, 1, 200}, 2, {1, 1, 1}));
  if (!oris3.isRowMajorSetContiguous()) {
    throw poprithms::test::error("oris3 should be row major set contiguous.");
  }

  Origins oris4({100, 2});
  oris4.insert(AllocId(6), Region::fromStripe({1, 1, 1, 300}, 3, {1, 2, 0}));
  oris4.insert(AllocId(6), Region::fromStripe({1, 1, 1, 300}, 3, {1, 2, 1}));
  if (oris4.isRowMajorSetContiguous()) {
    std::ostringstream oss;
    oss << "oris4 should not be row major set contiguous.\n"
        << "11.11.11.11.11.11.11.11.11.11.";
    throw poprithms::test::error(oss.str());
  }

  Origins oris5({10, 10});
  oris5.insert(AllocId(6), Region::fromStripe({1000, 5}, 0, {20, 980, 400}));
  if (!oris5.isRowMajorSetContiguous()) {
    std::ostringstream oss;
    oss << "oris5 should be row major set contiguous. Something like:";
    oss << ".....\n"
        << ".....\n"
        << "11111\n"
        << "11111\n"
        << ".....\n"
        << ".....\n";

    throw poprithms::test::error(oss.str());
  }

  Origins oris6({100, 2});
  oris6.insert(AllocId(6), Region::fromStripe({50, 4}, 1, {2, 2, 0}));
  if (oris6.isRowMajorSetContiguous()) {
    std::ostringstream oss;
    oss << "oris6 should not be row major set contiguous. Something like\n";
    oss << "111..\n"
        << "111..\n"
        << "111..\n";
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
