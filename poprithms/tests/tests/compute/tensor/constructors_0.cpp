// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::compute::host;
void testBasicConstructors() {

  // Construct from pointer:
  const double x{1.0};
  const auto a = Tensor::copyFloat64({}, &x);

  const std::array<double, 2> y{2.0, 3.0};
  const auto b = Tensor::copyFloat64({2}, y.data());

  // Construct from vector:
  const std::vector<double> z{4., 5., 6.};
  const auto c = Tensor::float64({3}, z);

  // Construct from move:
  const auto d = Tensor::float64({4}, {7., 8., 9., 10.});

  const auto abcd = concat({a.reshape({1}), b, c, d}, 0);
  const std::vector<double> expected{1., 2., 3, 4, 5, 6, 7, 8, 9, 10};

  if (abcd.getFloat64Vector() != expected) {
    throw poprithms::test::error("Unexpected result in construction test");
  }
}

void testRefConstructor() {
  double x{1.0};
  auto a = Tensor::refFloat64({}, &x);
  a      = a.add_(a);
  if (x - 2.0 != 0.) {
    throw poprithms::test::error("Unexpected result in testRefConstructor");
  }
}

void testBoolConstructor() {
  const auto t = Tensor::boolean({5}, {true, false, false, true, true});
  const auto x = t.getInt64Vector();
  if (x != std::vector<int64_t>{1, 0, 0, 1, 1}) {
    throw poprithms::test::error(
        "Unexpected result in testBoolConstructor (int vector)");
  }

  const auto v = t.getBooleanVector();
  if (v != std::vector<bool>{1, 0, 0, 1, 1}) {
    throw poprithms::test::error(
        "Unexpected result in testBoolConstructor (bool vector 1)");
  }

  const auto p = t.toFloat64().getBooleanVector();
  if (p != std::vector<bool>{1, 0, 0, 1, 1}) {
    throw poprithms::test::error(
        "Unexpected result in testBoolConstructor (bool vector 2)");
  }
}

void testScalarConstructors() {
  const auto a = Tensor::scalar(DType::Float16, 1.03);
  const auto b = Tensor::scalar(DType::Float64, 1.03);
  const auto c = a - b.toFloat16();
  c.assertAllEquivalent(Tensor::scalar(DType::Float16, 0.0));
}

void testInitializerListConstructors0() {
  const auto a = Tensor::scalar(DType::Float64, 7.);
  const auto b = Tensor::float64(7.);
  const auto c = Tensor::float64({}, {7});
  const auto d = Tensor::float64({}, {7.});
  const auto e = Tensor::float64({}, std::initializer_list<double>{7});
  const auto f = Tensor::float64({}, std::vector<double>{7});
  a.assertAllEquivalent(b);
  b.assertAllEquivalent(c);
  c.assertAllEquivalent(d);
  d.assertAllEquivalent(e);
  e.assertAllEquivalent(f);
  f.assertAllEquivalent(a);
}

void testCheckErrors0() {
  bool caught{false};
  try {
    Tensor::copyFloat64({1, 2}, nullptr);
  } catch (const poprithms::error::error &e) {
    std::cout << e.what();
    caught = true;
  }
  if (!caught) {
    std::ostringstream oss;
    throw poprithms::test::error(
        "Attempt to construct non-empty Tensor from nullptr should fail.");
  }
}

void testCheckErrors1() {
  bool caught{false};
  try {
    Tensor::float64({1, 2}, {1, 2, 3, 4});
  } catch (const poprithms::error::error &e) {
    std::cout << e.what();
    caught = true;
  }
  if (!caught) {
    std::ostringstream oss;
    throw poprithms::test::error(
        "Attempt to construct Tensor with Shape and n-elms mismatch.");
  }
}

} // namespace

int main() {
  testBasicConstructors();
  testRefConstructor();
  testBoolConstructor();
  testScalarConstructors();
  testInitializerListConstructors0();
  testCheckErrors0();
  testCheckErrors1();
  return 0;
}
