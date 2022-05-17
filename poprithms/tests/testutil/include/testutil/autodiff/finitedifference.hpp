// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_AUTODIFF_FINITEDIFFERENCE_HPP
#define TESTUTIL_AUTODIFF_FINITEDIFFERENCE_HPP

#include <poprithms/compute/host/tensor.hpp>

namespace poprithms {
namespace autodiff {
namespace finitedifference {

class Checker {

  using Tensor = poprithms::compute::host::Tensor;

public:
  /**
   *
   * The finite difference method for checking the correctness of
   * calculus-based gradients. See for example
   * https://cs231n.github.io/neural-networks-3/#gradcheck
   *
   * \param fwd: a function with 1 argument: a tensor of arbitrary shape. This
   *             function returns a scalar tensor (which we call the #loss).
   *
   * \param in0: the argument tensor to 'fwd'.
   *
   * \param in0grad: the expected gradient of #loss with respect to #in0.
   *                 It is computed externally using some logic which we
   *                 ultimately want to test using the finite difference
   *                 method.
   *
   * \param perturbationSize: the size of the perturbation which we will apply
   *                          to #in0 to estimate the change in #loss:
   *                          delta_loss = fwd(in0 + eps) - fwd(in0 - eps)
   *                          where
   *                          ||eps||_2 is proportional to perturbationSize.
   *
   * \param seed0: the random seed used to initialize the values of the
   *               perturbation to #in0.
   *
   * Let dc = delta loss using 'calculus'
   *     df = delta loss using the finite difference method.
   *
   * Then,
   *   relative error = |dc - df| / (max(dc, df) + #eps0).
   *
   * The test fails if relative error > #threshold.
   *
   * */
  static void check(const std::function<Tensor(const Tensor &)> &fwd,
                    const Tensor &in0,
                    const Tensor &in0grad,
                    double perturbationSize,
                    uint32_t seed0,
                    double eps0      = 1e-9,
                    double threshold = 1e-5);
};

} // namespace finitedifference
} // namespace autodiff
} // namespace poprithms

#endif
