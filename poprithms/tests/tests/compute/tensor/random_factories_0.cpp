// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {

using namespace poprithms::compute::host;

void testFloat() {

  const auto a0 = Tensor::uniformFloat32(0., 1.f, {}, 1011);
  const auto a1 = Tensor::uniformFloat32(0., 1.f, {}, 1011);
  const auto b0 = Tensor::uniformFloat32(0., 1.f, {}, 1012);
  const auto x0 = a0.getFloat32Vector()[0];
  const auto x1 = a1.getFloat32Vector()[0];
  const auto y0 = b0.getFloat32Vector()[0];

  if (x0 - x1 != 0.0f) {
    throw error("Tensors generated with same seed should be identical");
  }

  if (x0 - y0 == 0.0f) {
    throw error("Tensors generated with different seeds should be the same");
  }

  const auto c0 = Tensor::uniformFloat64(-10., 10., {2, 3, 5, 100}, 1013);
  if (c0.nelms_u64() != 2 * 3 * 5 * 100) {
    throw error("Incorrect number of elements in random Tensor");
  }
  auto vals = c0.getFloat64Vector();
  std::sort(vals.begin(), vals.end());
  if (vals[0] > -6. || vals.back() < 6.) {
    throw error("Statistical anomoly? Values probably don't follow uniform "
                "distribution.");
  }
}

void testInt0() {

  // WVP : probability greater than (1 - 2^(-40)).

  const auto a = Tensor::randomInt32(-1, 2, {10, 10}, 1011);
  const auto b = Tensor::randomInt32(-1, 2, {10, 10}, 1012);
  if (a.allEquivalent(b)) {
    throw error(
        "a and b were created with different seeds, should be different");
  }
  if (!a.reduceMax({}).allEquivalent(Tensor::int32(1))) {
    throw error("100 values sampled uniformly from {-1, 0, +1}, should be "
                "one which is +1 with VHP");
  }

  if (!a.reduceMin({}).allEquivalent(Tensor::int32(-1))) {
    throw error("100 values sampled uniformly from {-1, 0, +1}, should be "
                "one which is -1 with VHP");
  }
}

void testBool() {
  const auto a = Tensor::randomBoolean({100}, 10100)
                     .toInt16()
                     .reduceSum({})
                     .getInt16Vector()[0];

  if (a < 1 || a > 99) {
    throw error("100 coin flips, all came up heads? Unlikely. ");
  }
}
} // namespace

int main() {
  testFloat();
  testInt0();
  testBool();
}
