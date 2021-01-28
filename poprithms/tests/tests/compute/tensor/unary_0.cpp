// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;

void testMod0() {
  const auto a = Tensor::float16(5.5);
  const auto b = Tensor::float16(2.0);
  const auto c = a % b;
  c.assertAllEquivalent(Tensor::float16(1.5));
}
} // namespace

int main() {
  testMod0();
  return 0;
}
