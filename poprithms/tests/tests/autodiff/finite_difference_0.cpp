// Copyright 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/autodiff/testutil/finitedifference.hpp>
#include <poprithms/error/error.hpp>

namespace {

using poprithms::ndarray::Shape;
using namespace poprithms;
using namespace poprithms::autodiff;
using namespace poprithms::autodiff::automatic;
using poprithms::compute::host::OptionalTensor;
using poprithms::compute::host::Tensor;

using poprithms::autodiff::testutil::Checker;
void testLog0() {

  using namespace poprithms::compute::host;

  auto h = Tensor::float64(2.);
  OpIn<Tensor, OptionalTensor> gIn({h}, {h.log()}, {Tensor::float64(1.)});
  auto grads = LogAutodiffer::backpropagate(gIn);
  auto f     = [](const Tensor &t0) { return t0.log(); };

  double perturbationSize = 0.001;
  double eps0             = 1e-9;
  double threshold        = 1e-5;
  uint32_t seed           = 1011;

  Checker::check(
      f, h, grads.at(0).value(), perturbationSize, seed, eps0, threshold);

  // perturbation gets too large, second order gradients take effect.
  {
    bool caught{false};
    try {
      perturbationSize = 0.1;
      Checker::check(
          f, h, grads.at(0).value(), perturbationSize, seed, eps0, threshold);

    } catch (const poprithms::error::error &em) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch failure with large perturbation");
    }
  }
}

// 'h0' and 'h1' are the input tensors, around which the perturbation tests
// are performed. 'apply' is a binary function.
template <typename Bwd, typename Fwd>
void testBinaryElementwise0(Fwd &&apply, Tensor h0, Tensor h1) {

  struct AutodiffHelper {
  public:
    AutodiffHelper(const Shape &s0_, const Shape &s1_) : s0(s0_), s1(s1_) {}
    Shape inShape(InIndex i) const { return i == 0 ? s0 : s1; }
    static Tensor constantLike(const Tensor &t, double v) {
      return t.scalarOfSameType(v);
    }
    Shape s0;
    Shape s1;
  } autodiffHelper(h0.shape(), h1.shape());

  auto bp = [&autodiffHelper](const auto &gIn) {
    return Bwd::backpropagate(gIn, autodiffHelper);
  };

  const auto out = apply(h0, h1);

  // Gradient for reduceSum:
  const auto gradOut =
      Tensor::float64(1).expand(h0.shape().numpyBinary(h1.shape()));

  OpIn<Tensor, OptionalTensor> gIn({h0, h1}, {out}, {gradOut});
  auto grads         = bp(gIn);
  const auto grad_h0 = grads.at(0).value();
  const auto grad_h1 = grads.at(1).value();

  double perturbationSize = 0.0001;
  double eps0             = 1e-9;
  double threshold        = 1e-5;
  uint32_t seed           = 1011;

  // Check correctness for arg0.
  auto f0 = [&apply, h1](const Tensor &t0) {
    return apply(t0, h1).reduceSum();
  };
  Checker::check(f0, h0, grad_h0, perturbationSize, seed, eps0, threshold);

  // Check correctness for arg1.
  auto f1 = [&apply, h0](const Tensor &t1) {
    return apply(h0, t1).reduceSum();
  };
  Checker::check(f1, h1, grad_h1, perturbationSize, seed, eps0, threshold);

  // Verify that when the gradient is computed a bit different, the test
  // fails:
  bool caught{false};
  try {
    Checker::check(
        f0,
        h0,
        grad_h0.mul(
            Tensor::uniformFloat64(0, 0.001, h0.shape(), 1000).add(1)),
        perturbationSize,
        seed,
        eps0,
        threshold);

  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error when incorrect gradient used");
  }
}

void testBinaryOps0() {
  {
    const auto h0 = Tensor::float64({3, 1}, {1.5, -0.5, 1});
    const auto h1 = Tensor::float64({1, 2}, {3, -2});

    testBinaryElementwise0<DivAutodiffer>(
        [](const auto &a, const auto &b) { return a / b; }, h0, h1);

    testBinaryElementwise0<MulAutodiffer>(
        [](const auto &a, const auto &b) { return a * b; }, h0, h1);

    testBinaryElementwise0<SubAutodiffer>(
        [](const auto &a, const auto &b) { return a - b; }, h0, h1);
  }
  {

    // For h0^h1, h0 must be positive.
    const auto h0 = Tensor::float64({4, 1}, {1.5, 0.5, 1, 0.1});
    const auto h1 = Tensor::float64({1, 1, 3}, {3, -2, -0.5});

    testBinaryElementwise0<PowAutodiffer>(
        [](const auto &a, const auto &b) { return a.pow(b); }, h0, h1);
  }
}

void testMatMul0(Tensor h0, Tensor h1) {

  auto bp = [](const auto &gIn) {
    return MatMulAutodiffer::backpropagate(gIn);
  };

  auto apply = [](const auto &a, const auto &b) { return a.matmul(b); };

  const auto out = apply(h0, h1);

  // Gradient for reduceSum:
  const auto gradOut =
      Tensor::float64(1).expand(h0.shape().matmul(h1.shape()));

  OpIn<Tensor, OptionalTensor> gIn({h0, h1}, {out}, {gradOut});
  auto grads = bp(gIn);

  const auto grad_h0 = grads.at(0).value();
  const auto grad_h1 = grads.at(1).value();

  if (grad_h0.shape() != h0.shape() || grad_h1.shape() != h1.shape()) {
    throw poprithms::test::error("Gradients have incorrect shape");
  }

  double perturbationSize = 0.001;
  double eps0             = 1e-9;
  double threshold        = 1e-5;
  uint32_t seed           = 1011;

  // Check correctness for arg0.
  auto f0 = [&apply, h1](const Tensor &t0) {
    return apply(t0, h1).reduceSum();
  };

  Checker::check(f0, h0, grad_h0, perturbationSize, seed, eps0, threshold);

  // Check correctness for arg1.
  auto f1 = [&apply, h0](const Tensor &t1) {
    return apply(h0, t1).reduceSum();
  };
  Checker::check(f1, h1, grad_h1, perturbationSize, seed, eps0, threshold);
}

void testMatMuls0() {
  {
    const auto h0 = Tensor::uniformFloat64(-10, 10, {2, 1, 3, 2, 3}, 1011);
    const auto h1 = Tensor::uniformFloat64(-10, 10, {4, 1, 3, 4}, 1011);
    testMatMul0(h0, h1);
  }

  {
    const auto h0 = Tensor::uniformFloat64(-10, 10, {2, 3}, 1011);
    const auto h1 = Tensor::uniformFloat64(-10, 10, {3, 4}, 1011);
    testMatMul0(h0, h1);
  }

  {
    const auto h0 = Tensor::uniformFloat64(-10, 10, {2, 3}, 1011);
    const auto h1 = Tensor::uniformFloat64(-10, 10, {1, 1, 2, 1, 3, 4}, 1011);
    testMatMul0(h0, h1);
  }
}
} // namespace

int main() {
  testLog0();
  testBinaryOps0();
  testMatMuls0();
}
