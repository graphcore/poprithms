// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;

void testFromFloat64() {

  // 0, 0.2, 0.4, 0.6, 0.8, 1.0
  const auto f64 = Tensor::arangeFloat64(0., 1.1, 0.2);
  if (f64.dtype() != DType::Float64) {
    throw error("Expected initialized type to be Float64");
  }
  const auto f32 = f64.toFloat32();
  if (f32.dtype() != DType::Float32) {
    throw error("Expected type after cast to be Float32");
  }
  const auto f16 = f32.toFloat16();
  if (f16.dtype() != DType::Float16) {
    throw error("Expected after type to be Float16");
  }
  const auto i32 = f16.toInt32();
  if (i32.dtype() != DType::Int32) {
    throw error("Expected type after cast to be Int32");
  }
  const auto u32 = i32.toUnsigned32();
  if (u32.dtype() != DType::Unsigned32) {
    throw error("Expected type after cast to be Unsigned32");
  }
  if (u32.getInt32Vector() != std::vector<int>{0, 0, 0, 0, 0, 1}) {
    throw error("Error in the casting chain in cast_0");
  }
}
} // namespace

int main() {
  testFromFloat64();
  return 0;
}
