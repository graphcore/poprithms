// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void test1() {
  Region r0({6, 1, 8}, {{{{1, 1, 0}}}, {{{1, 0, 0}}}, {{{2, 2, 0}}}});
  const auto expanded = r0.expand({5, 6, 7, 8});

  if (!expanded.equivalent(Region(
          {5, 6, 7, 8},
          {{{{1, 0, 0}}}, {{{1, 1, 0}}}, {{{1, 0, 0}}}, {{{2, 2, 0}}}}))) {
    throw error("Failed to assert equivalence");
  }
}

void test0() {
  for (Region r0 : std::vector<Region>{
           {{}, {}},
           {{1}, {{{{1, 0, 0}}}}},
           {{1, 1, 1}, {{{{1, 0, 0}}}, {{{1, 0, 0}}}, {{{1, 0, 0}}}}}}) {
    const auto expanded = r0.expand({1, 2, 3});
    if (!expanded.equivalent(Region::createFull({1, 2, 3}))) {
      throw error("Failed in expand from 1 element Region");
    }
  }
}

} // namespace

int main() {
  test0();
  test1();
}
