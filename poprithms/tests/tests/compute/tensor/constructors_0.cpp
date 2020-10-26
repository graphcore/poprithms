// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {

using namespace poprithms::compute::host;
void testBasicConstructors() {

  // Construct from pointer:
  const double x{1.0};
  const auto a = Tensor::float64({}, &x);

  const std::array<double, 2> y{2.0, 3.0};
  const auto b = Tensor::float64({2}, y.data());

  // Construct from vector:
  const std::vector<double> z{4., 5., 6.};
  const auto c = Tensor::float64({3}, z);

  // Construct from move:
  const auto d = Tensor::float64({4}, {7., 8., 9., 10.});

  const auto abcd = concat({a.reshape({1}), b, c, d}, 0);
  const std::vector<double> expected{1., 2., 3, 4, 5, 6, 7, 8, 9, 10};

  if (abcd.getFloat64Vector() != expected) {
    throw error("Unexpected result in construction test");
  }
}

void testRefConstructor() {
  double x{1.0};
  auto a = Tensor::refFloat64({}, &x);
  a      = a.add_(a);
  if (x - 2.0 != 0.) {
    throw error("Unexpected result in testRefConstructor");
  }
}

void testBoolConstructor() {
  const auto t = Tensor::boolean({5}, {true, false, false, true, true});
  const auto x = t.getInt64Vector();
  if (x != std::vector<int64_t>{1, 0, 0, 1, 1}) {
    throw error("Unexpected result in testBoolConstructor (int vector)");
  }

  const auto v = t.getBooleanVector();
  if (v != std::vector<bool>{1, 0, 0, 1, 1}) {
    throw error("Unexpected result in testBoolConstructor (bool vector 1)");
  }

  const auto p = t.toFloat64().getBooleanVector();
  if (p != std::vector<bool>{1, 0, 0, 1, 1}) {
    throw error("Unexpected result in testBoolConstructor (bool vector 2)");
  }
}

void testScalarConstructors() {
  const auto a = Tensor::scalar(DType::Float16, 1.03);
  const auto b = Tensor::scalar(DType::Float64, 1.03);
  const auto c = a - b.toFloat16();
  c.assertAllEquivalent(Tensor::scalar(DType::Float16, 0.0));
}

} // namespace

int main() {
  testBasicConstructors();
  testRefConstructor();
  testBoolConstructor();
  testScalarConstructors();
  return 0;
}
