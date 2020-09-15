// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/error.hpp>

int main() {

  auto x = poprithms::ndarray::DType::Float64;
  if (poprithms::ndarray::nbytes(x) != 8) {
    throw poprithms::ndarray::error("Expected Float64 to have 8 bytes");
  }
  return 0;
}
