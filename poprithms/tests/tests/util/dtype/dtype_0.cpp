// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/dtype.hpp>

int main() {

  auto x = poprithms::ndarray::DType::Float64;
  if (poprithms::ndarray::nbytes(x) != 8) {
    throw poprithms::test::error("Expected Float64 to have 8 bytes");
  }
  return 0;
}
