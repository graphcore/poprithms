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
} // namespace

int main() { testLog0(); }
