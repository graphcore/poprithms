// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void assertComplement(const Region &r0, const DisjointRegions &expe) {
  if (!Region::equivalent(r0.getComplement(), expe)) {
    std::ostringstream oss;
    oss << "Failed in test assertComplement, for \nr0 = " << r0
        << ", \nexpected " << expe << ", \nobserved " << r0.getComplement();
    throw error(oss.str());
  }
}

void test0() {
  Region r0({10, 20}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}});
  DisjointRegions rs(
      {10, 20},
      {std::vector<Sett>{{{{1, 1, 1}}}, Sett::createAlwaysOn()},
       {{{{1, 1, 0}}}, {{{1, 1, 1}}}}});
  assertComplement(r0, rs);
}

} // namespace

int main() { test0(); }
