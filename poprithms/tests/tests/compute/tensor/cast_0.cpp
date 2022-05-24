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

void testRefToFloat16() {

  // A vector of uint16_t's which are bitwise float16's.
  auto d0 = Tensor::float32({2, 3}, {0.25, 0.5, 2.4, 2.7, 21, 100.5})
                .toFloat16()
                .getFloat16Vector_u16();

  auto d0_init = d0;

  // Create a tensor which wraps the pointer of d0, but does not manage it.
  const auto x0 = Tensor::refFloat16({2, 3}, d0.data());

  auto foo = x0.slice_({0, 0}, {1, 2}).add_(10);
  foo.assertAllEquivalent(Tensor::float32({1, 2}, {10.25, 10.5}).toFloat32());

  auto x1 = x0.copy();
  auto d1 = x0.mul(0).add(1).getFloat16Vector_u16();

  auto x2 = x0.reverse_(0);
  x0.updateRefFloat16(d1.data());

  x2.assertAllEquivalent(Tensor::float16(1).expand({2, 3}));

  if (d0.at(0) == d0_init.at(0) || d0.at(3) != d0_init.at(3)) {
    throw poprithms::test::error("Expect the first 2 elements of d0 to be "
                                 "changed, the others unchanged");
  }
}

void testRefToInt32() {

  std::array<int, 4> vals{2, 3, 4, 5};
  auto x = Tensor::refInt32({2, 2}, vals.data());
  x.reverse_(0).reverse_(1).mul_(Tensor::int32({2, 2}, {1, 0, 1, 0}));
  if (vals != std::array<int, 4>{0, 3, 0, 5}) {
    throw poprithms::test::error("Incorrect masking for values in ref array");
  }

  std::array<int, 4> vals2{1, 1, 1, 1};
  x.updateRefInt32(vals2.data());

  if (x.getInt64Vector() != std::vector<int64_t>{1, 1, 1, 1}) {
    throw poprithms::test::error(
        "Failed to update the pointer of the int32 tensor correctly");
  }
}
} // namespace

int main() {
  testFromFloat64();
  testScalarCreation();
  testGetPtrToOriginData();
  testRefToFloat16();
  testRefToInt32();
  return 0;
}
