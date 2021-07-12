// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testSqrt() {
  Tensor::float64(1024).expand_({2, 2}).sqrt_().assertAllEquivalent(
      Tensor::float64({2, 2}, {32., 32., 32., 32}));
  Tensor::float32(16).expand_({1, 1, 1}).sqrt().assertAllEquivalent(
      Tensor::float32({1, 1, 1}, {4}));
  Tensor::float16(9.0).sqrt().assertAllEquivalent(Tensor::float32(3.));
}

void testAbs() {
  Tensor::int32(-12).expand({3, 1}).abs().assertAllEquivalent(
      Tensor::float32({3, 1}, {12., 12., 12.}));
}

void testExp0() {
  // In this test, we use that 2.71^x < e^x < 2.72^x for  x in [1, 3).
  auto t0 = Tensor::uniformFloat64(1., 3., {100}, 1011);
  auto a  = Tensor::float64(2.71).pow(t0);
  auto b  = t0.exp();
  auto c  = Tensor::float64(2.72).pow(t0);
  auto N =
      ((b < a).to(DType::Int32) + (c < b).to(DType::Int32)).reduceSum({});
  N.assertAllEquivalent(Tensor::int32(0));
}

void testLog0() {

  // log(exp(x)) = x for all x.

  auto t0  = Tensor::uniformFloat64(-3., 3., {100}, 1011);
  auto out = t0.exp().log();
  out.assertAllClose(t0, 0.0, 1e-6);
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

void testMod() {
  Tensor::float16(6.5f).mod(3).assertAllEquivalent(Tensor::float32(0.5));
}

} // namespace

int main() {

  testSqrt();
  testAbs();
  testCeil();
  testFloor();
  testMod();
  testExp0();
  testLog0();

  return 0;
}
