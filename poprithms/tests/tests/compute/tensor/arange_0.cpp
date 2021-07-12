// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::compute::host;

void test0() {

  // 10 3 -4 ....
  const auto foo  = Tensor::arangeInt32(10, -25, -7);
  const auto vals = foo.getInt32Vector();
  uint64_t index  = 0;
  for (int i = 10; i > -25; i -= 7) {
    if (vals[index++] != i) {
      throw poprithms::test::error(
          "Error in first arange test, expected 10, 3, -4...");
    }
  }

  const auto bar = Tensor::arangeInt32(10, -25, +7);
  if (bar.nelms_u64() != 0) {
    throw poprithms::test::error(
        "when step and (stop - start) have different signs, the range "
        "should be empty");
  }

  const auto fRange = Tensor::arangeFloat16(-100., 100., 98.5);
  if (fRange.nelms_u64() != 3) {
    throw poprithms::test::error(
        "Should be exactly 3 elements in this float16 range (-100, "
        "-1.5, 97)");
  }
  const auto fVals = fRange.getFloat64Vector();
  if (fVals[1] + 1.5 != 0.0) {
    throw poprithms::test::error("Incorrect value (-100 + 98.5 = -1.5)");
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
