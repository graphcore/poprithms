// Copyright 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <sstream>

#include <testutil/autodiff/finitedifference.hpp>

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms;
using namespace poprithms::autodiff;
using namespace poprithms::autodiff::automatic;

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

  finitedifference::Checker::check(
      f, h, grads.at(0).value(), perturbationSize, seed, eps0, threshold);

  // perturbation gets too large, second order gradients take effect.
  {
    bool caught{false};
    try {
      perturbationSize = 0.1;
      finitedifference::Checker::check(
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

void testMul0() {

  using namespace poprithms::compute::host;

  const auto h0  = Tensor::float64({3, 1}, {1.5, 2, -1});
  const auto h1  = Tensor::float64({1, 2}, {3, -4});
  const auto out = h0 * h1;

  // Gradient for reduceSum:
  const auto gradOut = Tensor::float64(1).expand({3, 2});

  OpIn<Tensor, OptionalTensor> gIn({h0, h1}, {out}, {gradOut});
  auto grads         = MulAutodiffer::backpropagate(gIn);
  const auto grad_h0 = grads.at(0).value();
  const auto grad_h1 = grads.at(1).value();

  double perturbationSize = 0.0001;
  double eps0             = 1e-9;
  double threshold        = 1e-5;
  uint32_t seed           = 1011;

  // Check correctness for arg0.
  auto f0 = [h1](const Tensor &t0) { return (t0 * h1).reduceSum(); };
  finitedifference::Checker::check(
      f0, h0, grad_h0, perturbationSize, seed, eps0, threshold);

  // Check correctness for arg1.
  auto f1 = [h0](const Tensor &t1) { return (h0 * t1).reduceSum(); };
  finitedifference::Checker::check(
      f1, h1, grad_h1, perturbationSize, seed, eps0, threshold);

  // Verify that if the gradient computed is a bit different, the test fails:
  bool caught{false};
  try {
    finitedifference::Checker::check(
        f0,
        h0,
        grad_h0.mul(Tensor::uniformFloat64(0, 0.001, {3, 1}, 1000).add(1)),
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

} // namespace

int main() {
  testLog0();
  testMul0();
}
