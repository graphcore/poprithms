// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void assertSubtract(const Region &r0,
                    const Region &r1,
                    const DisjointRegions &expe) {
  if (!Region::equivalent(r0.subtract(r1), expe)) {
    std::ostringstream oss;
    oss << "Failed in test of Region::subtract. \n  " << r0 << ".subtract("
        << r1 << ") expected to be \n  " << expe << ", not \n  "
        << r0.subtract(r1);
    oss << ". This with " << r1 << ".getComplement() = \n  "
        << r1.getComplement();
    throw error(oss.str());
  }
}

void test0() {

  // r0       r1
  // 1111.    .....     1111.
  // 1111.  - ..111  =  11...
  // 1111.    ..111     11...
  // .....    ..111     .....

  const Region r0({{4, 5}, {{{{3, 1, 0}}}, {{{4, 1, 0}}}}});
  const Region r1({{4, 5}, {{{{3, 1, 1}}}, {{{3, 2, 2}}}}});

  assertSubtract(r0,
                 r1,
                 DisjointRegions({4, 5},
                                 std::vector<std::vector<Sett>>{
                                     {{{{1, 3, 0}}}, {{{4, 1, 0}}}},
                                     {{{{2, 2, 1}}}, {{{2, 3, 0}}}}}));

  assertSubtract(r1,
                 r0,
                 DisjointRegions({4, 5},
                                 std::vector<std::vector<Sett>>{
                                     {{{{3, 1, 1}}}, {{{1, 4, 4}}}},
                                     {{{{1, 3, 3}}}, {{{2, 3, 2}}}}}));

  assertSubtract(r0, r0, DisjointRegions::createEmpty({4, 5}));
}

void test1() {
  // 11.11.11.11.11.
  // .1..1..1..1..1.
  // 1..1..1..1..1..
  const Region r0({{10}, {{{{2, 1, 0}}}}});
  const Region r1({{10}, {{{{1, 2, 1}}}}});
  const Region r2({{10}, {{{{1, 2, 0}}}}});
  assertSubtract(r0, r1, {r2});
  assertSubtract(r0, r2, {r1});
}

} // namespace

int main() {
  test0();
  test1();
}
