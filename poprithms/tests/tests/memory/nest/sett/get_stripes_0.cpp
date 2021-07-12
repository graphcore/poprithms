// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

int main() {
  using namespace poprithms::memory::nest;
  Stripe s0(10, 10, 7);
  Sett p1{{s0, {12, 8, 0}}};
  if (p1.getStripes().size() != 1 || p1.getStripes()[0] != s0) {
    throw poprithms::test::error("Error in test of get_stripes");
  }
  return 0;
}
