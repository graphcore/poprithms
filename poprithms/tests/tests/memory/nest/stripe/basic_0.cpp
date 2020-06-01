// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/stripe.hpp>

int main() {

  using namespace poprithms::memory::nest;

  Stripe s0(10, 5, 17);
  if (s0.on() != 10 || s0.off() != 5 || s0.period() != 15 ||
      s0.phase() != 2) {
    throw error(
        "Error with Stripe construction, expected on=10, off=5, phase=2");
  }

  if (s0.nFullPeriods(0, 15) != 0 || s0.nFullPeriods(0, 17) != 1) {
    throw error("Error in Stripe test for nFullPeriods");
  }

  for (int64_t i : {1, 2, 3}) {
    for (int64_t j : {11, 12, 13}) {
      bool observed = s0.allOn(i, j);
      auto expected = i >= 2 && j <= 12;
      if (observed != expected) {
        throw error("Failure for allOn test of Stripe");
      }
    }
  }

  for (int64_t n : {0, 10}) {
    for (int64_t i : {11, 12, 13}) {
      for (int64_t j : {16, 17, 18}) {
        bool observed = s0.allOff(15 * n + i, 15 * n + j);
        auto expected = i >= 12 && j <= 17;
        if (observed != expected) {
          throw error("Failure for allOff test of Stripe");
        }
      }
    }
  }

  auto s1 = s0.getScaled(2);
  if (s1.on() != 20 || s1.phase() != 4) {
    throw error("Failed to scale correctly in Stripe test");
  }

  return 0;
}
