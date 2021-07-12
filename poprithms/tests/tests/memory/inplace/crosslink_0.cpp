// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/memory/nest/region.hpp>

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
      throw poprithms::test::error(
          "Error in testing move semantics of CrossLink");
    }

    if (!m1.isAliasing() || m1.isPureAliasing()) {
      throw poprithms::test::error(
          "Error testing aliasing of a modiying CrossLink");
    }
  }

  return 0;
}
