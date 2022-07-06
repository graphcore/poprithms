// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testNeg0(const DType &t) {

  std::vector<double> values{0.0, 1.0, -2.0, 3.0, -4.0, 5.0};
  std::vector<double> expected{-0.0, -1.0, 2.0, -3.0, 4.0, -5.0};
  auto a = Tensor::float64({2, 3}, values).to(t);
  auto b = a.neg();

  b.assertAllEquivalent(Tensor::float64({2, 3}, expected).to(t));
}

void testNeg1(const DType &t) {

  std::vector<double> values{0.0, 1.0, -2.0, 3.0, -4.0, 5.0};
  std::vector<double> expected{-0.0, -1.0, 2.0, -3.0, 4.0, -5.0};
  auto a = Tensor::float64({2, 3}, values).to(t);
  auto b = a.neg_();

  b.assertAllEquivalent(Tensor::float64({2, 3}, expected).to(t));
}

void testNeg0Aliases(const DType &t) {

  std::vector<double> values{0.0, 1.0, -2.0, 3.0, -4.0, 5.0};
  std::vector<double> expected{0.0, 1.0, -2.0, 3.0, -4.0, 5.0};
  auto a = Tensor::float64({2, 3}, values).to(t);
  auto b = a.neg();
  concat_({a, b}, 0).assertContainsNoAliases();

  a.assertAllEquivalent(Tensor::float64({2, 3}, expected).to(t));
}

void testNeg1Aliases(const DType &t) {

  std::vector<double> values{0.0, 1.0, -2.0, 3.0, -4.0, 5.0};
  std::vector<double> expected{-0.0, -1.0, 2.0, -3.0, 4.0, -5.0};
  auto a = Tensor::float64({2, 3}, values).to(t);
  auto b = a.neg_();
  concat_({a, b}, 0).assertContainsAliases();

  a.assertAllEquivalent(Tensor::float64({2, 3}, expected).to(t));
}

void testNeg0bool() {

  bool caught{false};
  try {
    const auto b = Tensor::boolean({2}, {true, false}).neg();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Expect: No Neg defined for bool.");
  }
}

void testNeg1bool() {

  bool caught{false};
  try {
    const auto b = Tensor::boolean({2}, {true, false}).neg_();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Expect: No Neg defined for bool.");
  }
}

void testNeg0Unsigned(const DType &t) {

  bool caught{false};
  try {
    const auto b = Tensor::float64({2}, {1, 2}).to(t).neg();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Expect: No Neg defined for unsigned.");
  }
}

void testNeg1Unsigned(const DType &t) {

  bool caught{false};
  try {
    const auto b = Tensor::float64({2}, {1, 2}).to(t).neg_();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Expect: No Neg defined for unsigned.");
  }
}

} // namespace

int main() {
  std::vector<DType> testTypes{DType::Int16,
                               DType::Int32,
                               DType::Int64,
                               DType::Float16,
                               DType::Float32,
                               DType::Float64};
  for (const auto &testType : testTypes) {
    testNeg0(testType);
    testNeg1(testType);
    testNeg0Aliases(testType);
    testNeg1Aliases(testType);
  }

  testNeg0bool();
  testNeg1bool();

  std::vector<DType> testUnsigned{DType::Unsigned8,
                                  DType::Unsigned16,
                                  DType::Unsigned32,
                                  DType::Unsigned64};
  for (const auto &testType : testUnsigned) {
    testNeg0Unsigned(testType);
    testNeg1Unsigned(testType);
  }

  return 0;
}
