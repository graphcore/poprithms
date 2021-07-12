// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void test1() {
  Region r0({6, 1, 8}, {{{{1, 1, 0}}}, {{{1, 0, 0}}}, {{{2, 2, 0}}}});
  const auto expanded = r0.expand({5, 6, 7, 8});

  if (!expanded.equivalent(Region(
          {5, 6, 7, 8},
          {{{{1, 0, 0}}}, {{{1, 1, 0}}}, {{{1, 0, 0}}}, {{{2, 2, 0}}}}))) {
    throw poprithms::test::error("Failed to assert equivalence");
  }
}

void test0() {
  for (Region r0 : std::vector<Region>{
           {{}, {}},
           {{1}, {{{{1, 0, 0}}}}},
           {{1, 1, 1}, {{{{1, 0, 0}}}, {{{1, 0, 0}}}, {{{1, 0, 0}}}}}}) {
    const auto expanded = r0.expand({1, 2, 3});
    if (!expanded.equivalent(Region::createFull({1, 2, 3}))) {
      throw poprithms::test::error("Failed in expand from 1 element Region");
    }
  }
}

void test2() {

  const Region r0({4, 1, 3},
                  {{{{1, 1, 0}}}, Sett::createAlwaysOn(), {{{1, 2, 0}}}});
  const Region r1({4, 1, 3},
                  {{{{1, 1, 0}}}, Sett::createAlwaysOn(), {{{2, 1, 2}}}});
  const Region r2(
      {4, 1, 3},
      {{{{1, 1, 1}}}, Sett::createAlwaysOn(), Sett::createAlwaysOn()});
  const DisjointRegions drs({4, 1, 3}, {r0, r1, r2});
  const auto expanded = drs.expand({5, 4, 6, 3});

  if (!expanded.equivalent(DisjointRegions::createFull({5, 4, 6, 3}))) {
    std::ostringstream oss;
    oss << "Expected equivalence : the 3 Regions partition the full 3-d shape"
        << ", and therefore when expanded, partition the 4-d shape.";
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
}
