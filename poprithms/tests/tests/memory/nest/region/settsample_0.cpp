// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

void assertSettSample(const Region &a,
                      const Region &where,
                      const DisjointRegions &expectedSample) {

  auto foo = a.settSample(where);
  if (!Region::equivalent(expectedSample, foo)) {
    std::ostringstream oss;
    oss << "in test for Region::equivalent, detected failure. "
        << "Expected the sample of a=" << a << " from where=" << where
        << " to be " << expectedSample << ", but it is " << foo << ".";
    throw poprithms::test::error(oss.str());
  }

  if (expectedSample.size() < foo.size()) {
    std::ostringstream oss;
    oss << "Got the correct expectedSample between " << a << " and " << where
        << " in this test for Region , "
        << " but the expected solution is more compact. ";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {
  Region a({10, 8}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}});
  Region where{{10, 8}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}}};
  Region expectedSample({5, 4},
                        {Sett::createAlwaysOn(), Sett::createAlwaysOn()});
  assertSettSample(a, where, {expectedSample});
}

void test1() {
  const Shape s({10000000, 8000000});
  Region a(s, {{{{1, 1, 0}}}, {{{1, 1, 0}}}});
  Region where{s, {{{{1, 1, 1}}}, {{{1, 1, 0}}}}};
  Region expectedSample({s.dim(0) / 2, s.dim(1) / 2},
                        {{{{0, 1, 0}}}, Sett::createAlwaysOn()});
  assertSettSample(a, where, {expectedSample});
}

} // namespace

int main() {

  using namespace poprithms::memory::nest;

  test0();
  test1();

  return 0;
}
