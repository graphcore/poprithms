// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;

void testSqrt() {
  Tensor::int32(1024).expand_({2, 2}).sqrt_().assertAllEquivalent(
      Tensor::float64({2, 2}, {32., 32., 32., 32}));
  Tensor::int32(5).expand_({1, 1, 1}).sqrt().assertAllEquivalent(
      Tensor::unsigned8({1, 1, 1}, {2}));
  Tensor::float16(9.0).sqrt().assertAllEquivalent(Tensor::float32(3.));
}

void testAbs() {
  Tensor::int32(-12).expand({3, 1}).abs().assertAllEquivalent(
      Tensor::float32({3, 1}, {12., 12., 12.}));
}

void testCeil() {
  Tensor::float32({1}, {1.5})
      .ceil()
      .assertAllEquivalent(Tensor::float32({1}, {2.}));
  Tensor::float32(1.5).ceil_().assertAllEquivalent(Tensor::float32(2.));
  Tensor::unsigned16(12).ceil().assertAllEquivalent(Tensor::unsigned16(12));
}

void testFloor() {
  Tensor::float64(1.5).floor().assertAllEquivalent(Tensor::float32(1.f));
  Tensor::float16(1.5).floor_().assertAllEquivalent(Tensor::float32(1.));
  Tensor::int8(15).floor().assertAllEquivalent(Tensor::int8(15));
  Tensor::int64(3).floor().floor_().floor().floor_().assertAllEquivalent(
      Tensor::int64(3).ceil().ceil_().ceil());
}

} // namespace

int main() {

  testSqrt();
  testAbs();
  testCeil();
  testFloor();

  return 0;
}
