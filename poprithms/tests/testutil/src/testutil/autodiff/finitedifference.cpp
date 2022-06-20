// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <testutil/autodiff/finitedifference.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace autodiff {
namespace finitedifference {

using poprithms::compute::host::Tensor;

void Checker::check(const std::function<Tensor(const Tensor &)> &fwd,
                    const Tensor &in0,
                    const Tensor &in0grad,
                    const double perturbationSize,
                    const uint32_t seed0,
                    const double eps0,
                    const double threshold) {

  using namespace poprithms::compute::host;

  const uint64_t nRuns = 5;

  // How much does the loss change with the perturbation? This vector collects
  // the values for each random perturbation, as computed by finite
  // difference.
  std::vector<double> finiteMethod;

  // According to the gradient in0grad, what do we expect the change in loss
  // caused by the perurbation to be? This (obviously) ignores all but the
  // first order derivative.
  std::vector<double> expected;

  // A random vector of norm 1:
  auto purePerturbation =
      Tensor::uniformFloat64(-1, 1, in0.shape(), seed0).mul(perturbationSize);
  purePerturbation = purePerturbation.divide(purePerturbation.l2norm());

  const auto in0gradNormalized = in0grad.divide(in0grad.l2norm());

  auto phiPure = [](uint64_t r) { return r / (nRuns - 1.); };

  for (uint64_t run = 0; run < nRuns; ++run) {

    // The perturbation we use is a linear combination of the "pure"
    // perturbation and the gradient.
    // When,
    //  run = 0        : perturbation is the direction of the gradient.
    //  run = nRuns -1 : perturbation is the random vector.
    //
    //
    // Motivation for the combination:
    //   1) a pure random vector might almost always result in changes in
    //      loss which are too small.
    //   2) taking the perturbation to always be just the computed gradient
    //      ignores all other directions.
    //
    auto perturbation = in0gradNormalized.mul(1 - phiPure(run)) +
                        purePerturbation.mul(phiPure(run));

    perturbation = perturbation.mul(perturbationSize * perturbation.l2norm());
    const auto lossPlus = fwd(in0 + perturbation);

    if (lossPlus.nelms() != 1) {
      throw poprithms::test::error(
          "The method 'fwd' must produce a tensor with 1 element");
    }
    const auto lossMinus = fwd(in0 - perturbation);
    const auto deltaLoss = (lossPlus - lossMinus);
    finiteMethod.push_back(deltaLoss.getFloat64(0));

    const auto expectedDeltaLoss =
        (in0grad * perturbation).reduceSum().getFloat64(0) * 2.0;
    expected.push_back(expectedDeltaLoss);
  }

  std::vector<double> relErrs;

  for (uint64_t i = 0; i < finiteMethod.size(); ++i) {

    const auto a = finiteMethod[i];
    const auto b = expected[i];

    const auto relErr =
        std::abs(a - b) / (std::max(std::abs(a), std::abs(b)) + eps0);

    relErrs.push_back(relErr);
  }

  for (uint64_t i = 0; i < finiteMethod.size(); ++i) {

    if (relErrs[i] > threshold) {
      std::ostringstream oss;
      oss << "Finite difference test failed. ";

      oss << "\nRun #" << i << " (of " << nRuns << ")."
          << "\nFraction of perturbation in direction of gradient: "
          << 1 - phiPure(i) << ".";
      oss << "\n    Delta loss with FD method          : " << finiteMethod[i]
          << "\n    Delta loss using provided gradient : " << expected[i]
          << "\n    Relative error                     : " << relErrs[i]
          << '\n';
      throw poprithms::test::error(oss.str());
    }
  }
}
} // namespace finitedifference
} // namespace autodiff
} // namespace poprithms
