// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testReciprocal0(const DType &t) {

  auto a = Tensor::arangeFloat64(1.0, 7.0, 1.0).reshape({2, 3}).to(t);
  auto b = a.reciprocal();

  b.assertAllEquivalent(
      Tensor::float64({2, 3},
                      {1. / 1., 1. / 2., 1. / 3., 1. / 4., 1. / 5., 1. / 6.})
          .to(t));
}

void testReciprocal1(const DType &t) {

  auto a = Tensor::arangeFloat64(1.0, 7.0, 1.0).reshape({2, 3}).to(t);
  auto b = a.reciprocal_();

  b.assertAllEquivalent(
      Tensor::float64({2, 3},
                      {1. / 1., 1. / 2., 1. / 3., 1. / 4., 1. / 5., 1. / 6.})
          .to(t));
}

void testReciprocal0Aliases(const DType &t) {

  auto a = Tensor::arangeFloat64(1.0, 7.0, 1.0).reshape({2, 3}).to(t);
  auto b = a.reciprocal();
  concat_({a, b}, 0).assertContainsNoAliases();

  a.assertAllEquivalent(
      Tensor::float64({2, 3}, {1., 2., 3., 4., 5., 6.}).to(t));
}

void testReciprocal1Aliases(const DType &t) {

  auto a = Tensor::arangeFloat64(1.0, 7.0, 1.0).reshape({2, 3}).to(t);
  auto b = a.reciprocal_();
  concat_({a, b}, 0).assertContainsAliases();

  a.assertAllEquivalent(
      Tensor::float64({2, 3},
                      {1. / 1., 1. / 2., 1. / 3., 1. / 4., 1. / 5., 1. / 6.})
          .to(t));
}

void testReciprocal0bool() {

  bool caught{false};
  try {
    const auto b = Tensor::boolean({2}, {true, false}).reciprocal();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Expect: No Reciprocal defined for bool.");
  }
}

void testReciprocal1bool() {

  bool caught{false};
  try {
    const auto b = Tensor::boolean({2}, {true, false}).reciprocal_();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Expect: No Reciprocal defined for bool.");
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
    testReciprocal0(testType);
    testReciprocal1(testType);
    testReciprocal0Aliases(testType);
    testReciprocal1Aliases(testType);
  }

  testReciprocal0bool();
  testReciprocal1bool();

  return 0;
}
