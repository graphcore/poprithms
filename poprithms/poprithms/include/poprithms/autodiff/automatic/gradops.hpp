// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPOPS_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPOPS_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::ndarray::Shape;

/**
 * Helper template class for differentiating log (natural base).
 *
 * out = log(in)                        (1)
 *
 * dLoss/dIn = dLoss/dOut * dOut/dIn    (2)
 *           = gradient-of-out / in.    (3)
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

/**
 * Helper template class for differentiating an add op with numpy-broadcasting
 * support.
 *
 * (1) out = in0 + in1
 *
 * (2) dLoss/dIn0 = dLoss/dOut.reduceSum(in0.shape)
 * (3) dLoss/dIn1 = dLoss/dOut.reduceSum(in1.shape)
 * */
class AddAutodiffer {

public:
  /**
   * Neither of the inputs to the add (in0 and in1) are required in equations
   * (2) and (3).
   * */
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }

  /**
   * The output of the add (out) is not required in equations (2) and (3).
   * */
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  /**
   * Equations (2) and (3).
   * */
  template <typename Tensor, typename OptionalTensor>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn,
                const Shape &in0,
                const Shape &in1) {
    auto grad = gIn.gradOfOutput(0);
    auto g0   = grad.reduceSum(in0);
    auto g1   = grad.reduceSum(in1);
    return {OptionalTensor(g0), OptionalTensor(g1)};
  }
};

/**
 * Helper template class for differentiating a mul op with numpy-broadcasting
 * support.
 *
 * (1) out = in0 * in1
 *
 * (2) dLoss/dIn0 = (dLoss/dOut * in1).reduceSum(in0.shape)
 * (3) dLoss/dIn1 = (dLoss/dOut * in0).reduceSum(in1.shape)
 * */
class MulAutodiffer {

public:
  static std::vector<InIndex> autodiffRequiredIns() { return {0, 1}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  /**
   * Equations (2) and (3).
   * */
  template <typename Tensor, typename OptionalTensor>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn) {
    auto grad = gIn.gradOfOutput(0);
    auto in0  = gIn.input(0);
    auto in1  = gIn.input(1);
    auto g0   = (grad * in1).reduceSum(in0.shape());
    auto g1   = (grad * in0).reduceSum(in1.shape());

    return {OptionalTensor(g0), OptionalTensor(g1)};
  }
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
