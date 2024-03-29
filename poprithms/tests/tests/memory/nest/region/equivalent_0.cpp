// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {
using namespace poprithms::memory::nest;
void assertEquivalence(const DisjointRegions &a,
                       const DisjointRegions &b,
                       bool equiv) {
  if (Region::equivalent(a, b) != equiv) {
    std::ostringstream oss;
    oss << "Failed in assertEquivalence test for " << a << " and " << b
        << " : expected equiv=" << equiv;
    throw poprithms::test::error(oss.str());
  }
}
} // namespace

int main() {
  assertEquivalence(DisjointRegions::createEmpty({5, 6, 7}),
                    DisjointRegions::createEmpty({5, 6, 7}),
                    true);

  assertEquivalence(DisjointRegions::createEmpty({}),
                    DisjointRegions::createEmpty({}),
                    true);

  DisjointRegions a(
      {100, 200},
      {{{100, 200}, {Sett::createAlwaysOn(), Sett::createAlwaysOn()}}});

  DisjointRegions b({100, 200},
                    {{{100, 200}, {Sett::createAlwaysOn(), {{{1, 1, 1}}}}},
                     {{100, 200}, {Sett::createAlwaysOn(), {{{1, 1, 0}}}}}});

  DisjointRegions c({100, 200},
                    {{{100, 200}, {Sett::createAlwaysOn(), {{{1, 3, 1}}}}},
                     {{100, 200}, {Sett::createAlwaysOn(), {{{1, 1, 0}}}}}});

  DisjointRegions d({100, 200},
                    {{{100, 200}, {{{{1, 2, 0}}}, Sett::createAlwaysOn()}},
                     {{100, 200}, {{{{1, 2, 1}}}, Sett::createAlwaysOn()}},
                     {{100, 200}, {{{{1, 2, 2}}}, Sett::createAlwaysOn()}}});

  assertEquivalence(a, a, true);
  assertEquivalence(b, d, true);
  assertEquivalence(a, c, false);
  assertEquivalence(b, c, false);
}
