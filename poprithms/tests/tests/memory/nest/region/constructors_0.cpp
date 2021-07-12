// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

int main() {

  using namespace poprithms::memory::nest;

  const Region r0({3, 5}, {Sett::createAlwaysOn(), {{{1, 1, 0}}}});
  const Region r1({4, 6}, {Sett::createAlwaysOn(), {{{1, 1, 0}}}});
  // copy constructor
  auto r2 = r0;
  // copy operator
  r2 = r1;
  // move constructor
  const auto r3 = std::move(r2);
  const auto r4 = Region::fromStripe({4, 6}, 1, {1, 1, 0});
  if (!r4.equivalent(r1)) {
    throw poprithms::test::error(
        "Stripe constructor gives unexpected Region");
  }
  if (r4.equivalent(r0)) {
    throw poprithms::test::error(
        "r4 compared equvalent to r0, in constructro test");
  }

  const auto r5 = Region::fromBounds({10}, {3}, {6});
  if (!r5.equivalent({{10}, {{{{6 - 3, 10 - (6 - 3), 3}}}}})) {
    std::cout << r5 << std::endl;
    throw poprithms::test::error(
        "Incorrect region constructed by fromBounds");
  }
}
