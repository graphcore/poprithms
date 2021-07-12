// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void testReduce() {

  //
  // 1.1.1.      .1.1.1
  // ......      ......
  // ......  and ......
  // 1.1.1.      .1.1.1
  // ......      ......
  //
  DisjointRegions foo0({7, 17},
                       {
                           {{{{1, 1, 0}}}, {{{1, 2, 0}}}}, // reg0
                           {{{{1, 1, 1}}}, {{{1, 2, 1}}}}  // reg1
                       });

  // Test 0
  const auto reduced0 = foo0.reduce({7, 1});
  if (!Region::equivalent(reduced0, DisjointRegions::createFull({7, 1}))) {
    throw poprithms::test::error("Unexpected result in testReduce, test 0");
  }

  // Test 1
  const auto reduced1 = foo0.reduce({1, 17});
  if (!Region::equivalent(
          reduced1,
          DisjointRegions({1, 17}, {{{{{1, 1, 0}}}, {{{2, 1, 0}}}}}))) {
    throw poprithms::test::error("Unexpected result in testReduce, test 1");
  }

  //   Test 2
  DisjointRegions foo2({7, 17},
                       {
                           {{{{1, 1, 0}}}, {{{1, 2, 0}}}}, // reg0
                           {{{{1, 1, 0}}}, {{{1, 2, 1}}}}  // reg1
                       });
  const auto reduced2 = foo2.reduce({7, 1});
  if (!Region::equivalent(
          reduced2,
          DisjointRegions({7, 1},
                          {Region::fromStripe({7, 1}, 0, {1, 1, 0})}))) {
    throw poprithms::test::error("Unexpected result in testReduce, test 2");
  }

  //  Test 3
  std::cout << "test 3 " << std::endl;
  DisjointRegions foo3({7, 17},
                       {
                           {{{{1, 1, 0}}}, {{{2, 2, 0}}}}, // reg0
                           {{{{1, 1, 0}}}, {{{2, 2, 1}}}}  // reg1
                       });

  const auto reduced3 = foo3.reduce({1, 17});
  if (!Region::equivalent(
          reduced3,
          DisjointRegions({1, 17}, {{{{{1, 1, 0}}}, {{{3, 1, 0}}}}}))) {
    throw poprithms::test::error("Unexpected result 3");
  }
}

} // namespace

int main() { testReduce(); }
