// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

int main() {
  using namespace poprithms::compute::host;

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
