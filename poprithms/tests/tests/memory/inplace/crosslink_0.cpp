// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/memory/inplace/error.hpp>

int main() {
  using namespace poprithms::memory::inplace;

  {
    auto m1 = CrossLink::modifies(0, 0);
    auto m2 = m1;
    auto m3 = m1;
    m3      = m2;
    m3      = std::move(m2);
    CrossLink m4(std::move(m3));
    if (!m1.isModifying() || !m4.isModifying()) {
      throw error("Error in testing move semantics of CrossLink");
    }

    if (!m1.isAliasing() || m1.isPureAliasing()) {
      throw error("Error testing aliasing of a modiying CrossLink");
    }
  }

  auto a = CrossLink::pureAliases(0, 0);

  auto u  = CrossLink::uses(0, 0, std::make_unique<IdentityRegsMap>());
  auto u2 = CrossLink::uses(0, 0, std::make_unique<IdentityRegsMap>());
  if (u != u2) {
    throw error("Failed in comparison test of CrossLinks");
  }
  const DisjointRegions dr({5, 5},
                           {Region::fromBounds({5, 5}, {1, 1}, {4, 4})});
  auto out = u.fwd(dr);
  if (!out.equivalent(dr)) {
    std::ostringstream oss;
    oss << "Failed CrossLink::uses test with IdentityRegsMap";
    throw error(oss.str());
  }

  return 0;
}
