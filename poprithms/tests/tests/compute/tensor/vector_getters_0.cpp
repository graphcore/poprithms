// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
void test0() {
  using namespace poprithms::compute::host;
  const auto i32 = Tensor::arangeInt32(-10, 10, 1);
  //-10 .... 9
  auto vals = i32.getFloat32Vector();
  for (auto &x : vals) {
    x *= 0.5f;
  }
  if (vals.size() != 20) {
    throw poprithms::test::error(
        "This arange should produce exactly 20 values");
  }
  for (uint64_t i = 0; i < 20; ++i) {
    if (vals[i] - (-5.f + 0.5f * static_cast<float>(i)) != 0.0f) {
      throw poprithms::test::error("An error in test0");
    }
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
