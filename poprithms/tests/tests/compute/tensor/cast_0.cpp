// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testFromFloat64() {

  // 0, 0.2, 0.4, 0.6, 0.8, 1.0
  const auto f64 = Tensor::arangeFloat64(0., 1.1, 0.2);
  if (f64.dtype() != DType::Float64) {
    throw poprithms::test::error("Expected initialized type to be Float64");
  }
  const auto f32 = f64.toFloat32();
  if (f32.dtype() != DType::Float32) {
    throw poprithms::test::error("Expected type after cast to be Float32");
  }
  const auto f16 = f32.toFloat16();
  if (f16.dtype() != DType::Float16) {
    throw poprithms::test::error("Expected after type to be Float16");
  }
  const auto i32 = f16.toInt32();
  if (i32.dtype() != DType::Int32) {
    throw poprithms::test::error("Expected type after cast to be Int32");
  }
  const auto u32 = i32.toUnsigned32();
  if (u32.dtype() != DType::Unsigned32) {
    throw poprithms::test::error("Expected type after cast to be Unsigned32");
  }
  if (u32.getInt32Vector() != std::vector<int>{0, 0, 0, 0, 0, 1}) {
    throw poprithms::test::error("Error in the casting chain in cast_0");
  }
}

void testScalarCreation() {

  for (auto t : {DType::Boolean, DType::Unsigned16}) {

    bool caught{false};
    try {
      Tensor::scalar(t, -1.0);
    } catch (const poprithms::error::error &) {
      std::ostringstream oss;
      caught = true;
    }
    if (!caught) {
      std::ostringstream oss;
      oss << "Failed to catch error creating " << t
          << " from negative double";
      throw poprithms::test::error(oss.str());
    }
  }
}

void testGetPtrToOriginData() {
  const auto t0 = Tensor::arangeFloat32(5., 10., 1.);

  {
    const auto vptr = t0.getPtrToOriginData();
    const auto fptr = static_cast<const float *>(vptr);
    for (uint64_t i = 0; i < t0.nelms_u64(); ++i) {
      if (fptr[i] != t0.getFloat32(i)) {
        throw poprithms::test::error("Failure in getPtrToOriginData");
      }
    }
  }

  bool caught{false};
  auto c = t0.subSample_(Stride(2), Dimension(0));
  try {
    c.getPtrToOriginData();
  } catch (const poprithms::error::error &err) {
    const std::string w = err.what();
    if (w.find("not contiguous") != std::string::npos) {
      caught = true;
    }
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error stating non contiguous Tensors can't have "
        "getPtrToOriginData called on them.");
  }

  auto foo = t0.getPtrToOriginData(3);
  if (*static_cast<const float *>(foo) != 5. + 3) {
    throw poprithms::test::error("Failed in test of getPtrToOriginData when "
                                 "there's a non-zero rowMajorIndex offset");
  }
}
} // namespace

int main() {
  testFromFloat64();
  testScalarCreation();
  testGetPtrToOriginData();
  return 0;
}
