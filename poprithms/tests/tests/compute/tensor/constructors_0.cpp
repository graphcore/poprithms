// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>

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

void testScalarConstructors0() {
  // no rounding error:
  {
    const auto a = Tensor::scalar(DType::Float32, 1.7f);
    const auto b = Tensor::safeScalar(DType::Float32, 1.7f);
  }

  // rounding error but it's not detected (correctly)
  { const auto b = Tensor::scalar(DType::Float32, 1.7); }

  bool caught = false;
  try {
    const auto b = Tensor::safeScalar(DType::Float32, 1.7);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch rounding error, 1.7 != "
                                 "1.7f in testScalarConstructors0");
  }

  caught = false;
  try {
    int64_t large = (1LL << 58) + 1;
    const auto c  = Tensor::safeScalar(DType::Int64, large);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch construction of large integer tensor from double in "
        "testScalarConstructors0");
  }
}

void testScalarConstructors1() {

  const auto a = Tensor::scalar(DType::Float16, 1.125);
  const auto b = Tensor::scalar(DType::Float64, 1.125);

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
    caught = true;
  }
  if (!caught) {
    std::ostringstream oss;
    throw poprithms::test::error(
        "Attempt to construct Tensor with Shape and n-elms mismatch.");
  }
}

void testTemplateConstructors0() {
  auto d = Tensor::tensor<uint64_t>({2}, {199, 8001});
  auto e = Tensor::unsigned64({2}, {199, 8001});
  assert(d.dtype() == e.dtype());
  d.assertAllEquivalent(e);

  std::vector<std::pair<Tensor, DType>> tensorsAndExpectedTypes{
      {Tensor::tensor<uint32_t>({}, {0}), DType::Unsigned32},
      {Tensor::tensor<uint16_t>({}, {0}), DType::Unsigned16},
      {Tensor::tensor<uint8_t>({}, {0}), DType::Unsigned8},
      {Tensor::tensor<int64_t>({}, {0}), DType::Int64},
      {Tensor::tensor<int32_t>({}, {0}), DType::Int32},
      {Tensor::tensor<int16_t>({}, {0}), DType::Int16},
      {Tensor::tensor<int8_t>({}, {0}), DType::Int8},
      {Tensor::tensor<float>({}, {0}), DType::Float32},
      {Tensor::tensor<double>({}, {0}), DType::Float64}};

  // What I was most interested in here was that there were no linking errors
  // because the implementations of the template class are not exposed. But we
  // confirm that the ranks and types are created as expected too.
  for (auto t : tensorsAndExpectedTypes) {
    if (t.first.rank_u64() != 0 || t.first.dtype() != t.second) {
      throw poprithms::test::error("Incorrect rank or type");
    }
  }
}

template <typename T> void lowestScalarTest0() {
  auto foo = Tensor::lowestScalar(poprithms::ndarray::get<T>())
                 .getNativeCharVector();
  T v = *reinterpret_cast<T *>(&foo[0]);
  if (v != std::numeric_limits<T>::lowest()) {
    throw poprithms::test::error("Failed in assert for lowest scalar");
  }
}

void testLowestScalar0() {
  lowestScalarTest0<double>();
  lowestScalarTest0<uint8_t>();
  lowestScalarTest0<int64_t>();
  lowestScalarTest0<bool>();

  auto lowestFloat16 = Tensor::lowestScalar(DType::Float16);
  auto asFloat64     = lowestFloat16.to(DType::Float64);

  // Note that we don't do arithmetic in float16, so for example
  // lowest + 1 != lowest (for true float16 they should be equal).
  if (!std::isinf(lowestFloat16.mul(2).toFloat32().getFloat32(0))) {
    throw poprithms::test::error("Exected lowest * 2 to be (-)infinite");
  }
}

} // namespace

int main() {
  testBasicConstructors();
  testRefConstructor();
  testBoolConstructor();
  testScalarConstructors0();
  testScalarConstructors1();
  testInitializerListConstructors0();
  testCheckErrors0();
  testCheckErrors1();
  testTemplateConstructors0();
  testLowestScalar0();
  return 0;
}
