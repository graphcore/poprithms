// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>

namespace {

using namespace poprithms::ndarray;
void testDType0() {
  auto x = poprithms::ndarray::DType::Float64;
  if (poprithms::ndarray::nbytes(x) != 8) {
    throw poprithms::test::error("Expected Float64 to have 8 bytes");
  }
}

void testTensorInfo0() {
  TensorInfo x0(Shape({4, 4}), DeviceId(0), DType::Float16);
  TensorInfo expected0(Shape({4, 4}), DeviceId(0), DType::Float16);

  auto x1 = x0.withShape({3, 3})
                .withDeviceId(DeviceId(1))
                .withDType(DType::Float32);

  TensorInfo expected1({3, 3}, 1, DType::Float32);

  if (expected0 != x0 || expected1 != x1 || expected0 == x1) {
    std::ostringstream oss;
    oss << "Expected x0 : " << expected0 << "\nObserved x0 : " << x0
        << "\nExpected x1 : " << expected1 << "\nObserved x1 : " << x1;
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {
  testTensorInfo0();
  testDType0();
  return 0;
}
