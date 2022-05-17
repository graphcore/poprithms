// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPOPS_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPOPS_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

/**
 * Helper template class for differentiating log (natural base).
 *
 * out = log(in)                        (1)
 *
 * dLoss/dIn = dLoss/dOut * dOut/dIn    (2)
 *          = gradient-of-out / in.     (3)
 * */
class LogAutodiffer {
public:
  /**
   * Input of log is required to compute its gradient (at least, for this
   * implementation of log differentiation).
   *  */
  static std::vector<InIndex> autodiffRequiredIns() { return {0}; }

  /** Output of log is not required to compute its gradient. */
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }

  /** A non-zero gradient does propagate through log. */
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  /**
   * Compute gradient of input.
   *
   * \tparam Tensor a tensor class for which the binary operator/ is defined
   *                and returns a tensor.
   *
   * \tparam OptionalTensor a class with a subset of the API of
   *                       std::optional<Tensor>.
   * */
  template <typename Tensor, typename OptionalTensor>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn) {

    const auto inputToLog   = gIn.input(0);
    const auto gradOfOutput = gIn.gradOfOutput(0);

    return {OptionalTensor(gradOfOutput / inputToLog)};
  }
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
