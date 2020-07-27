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
    oss << "Failed in test of Region::subtract. " << r0 << ".subtract(" << r1
        << ") expected to be " << expe << ", not " << r0.subtract(r1);
    throw error(oss.str());
  }
}

void test0() {

  // In dimension 1:
  // 11111111..  r0
  // ..11111111  r1
  //
  // And in dimension 2:
  // 111111111111111.....  r0
  // .....111111111111111  r1
  //
  const Region r0({{10, 20}, {{{{8, 2, 0}}}, {{{15, 5, 0}}}}});
  const Region r1({{10, 20}, {{{{8, 2, 2}}}, {{{15, 5, 5}}}}});
  assertSubtract(r0, r1, {{{10, 20}, {{{{2, 8, 0}}}, {{{5, 15, 0}}}}}});
  assertSubtract(r1, r0, {{{10, 20}, {{{{2, 8, 8}}}, {{{5, 15, 15}}}}}});
  assertSubtract(r0, r0, DisjointRegions::createEmpty({10, 20}));
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
