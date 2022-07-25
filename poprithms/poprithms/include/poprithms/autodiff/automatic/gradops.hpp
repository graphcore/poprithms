// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPOPS_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPOPS_HPP

#include <sstream>

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

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
   *                        std::optional<Tensor>.
   * */
  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const Args &...) {

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
   *
   * \tparam OpHelper: An object with a method #inShape(InIndex) for getting
   *                   the input shapes of the add op being differentiated.
   * */
  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const OpHelper &op) {
    auto grad = gIn.gradOfOutput(0);
    auto g0   = grad.reduceSum(op.inShape(InIndex(0)));
    auto g1   = grad.reduceSum(op.inShape(InIndex(1)));
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
  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const Args &...) {
    auto grad = gIn.gradOfOutput(0);
    auto in0  = gIn.input(0);
    auto in1  = gIn.input(1);
    auto g0   = (grad * in1).reduceSum(in0.shape());
    auto g1   = (grad * in0).reduceSum(in1.shape());
    return {OptionalTensor(g0), OptionalTensor(g1)};
  }
};

/**
 * Differentiation through a matrix multiplication (matmul). The matmul can
 * follow numpy broadcasting rules except that the inputs must be rank-2 or
 * greater.
 *
 * Example. Consider C = A * B. Where the tensors have shapes:
 *
 * C : (3,4,5,6,7)
 * A : (1,5,6,10)
 * B : (3,4,1,10,7).
 *
 * The gradients of A and B in terms of the gradient of C are easily shown to
 * be:
 *
 * (1)  dA = matmul(dC, B.transpose).reduceSum(A.shape())
 * (2)  DB = matmul(A.transpose, dC).reduceSum(B.shape())
 *
 * where dC is the gradient of C, and X.transpose is X with the final 2
 * dimensions swapped.
 * */
class MatMulAutodiffer {

public:
  /**
   * Both the inputs are required (see (1) and (2)).
   * */
  static std::vector<InIndex> autodiffRequiredIns() { return {0, 1}; }

  /**
   * The output of the matmul (C) is not required to compute the gradients of
   * A and B.
   * */
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const Args &...) {

    auto A  = gIn.input(0);
    auto B  = gIn.input(1);
    auto dC = gIn.gradOfOutput(0);

    auto dA = dC.matmul(dimShuffleFinalTwo(B)).reduceSum(A.shape());
    auto dB = dimShuffleFinalTwo(A).matmul(dC).reduceSum(B.shape());

    return {dA, dB};
  }

private:
  // A.transpose == dimShuffleFinalTwo(A).
  template <typename Tensor>
  static Tensor dimShuffleFinalTwo(const Tensor &t) {
    return t.dimShuffle(
        poprithms::util::Permutation::reverseFinalTwo(t.rank_u64()));
  }
};

/**
 * Differentiate,
 * (1)    out = numerator / denominator.
 *
 * Ignoring numpy broadcast for now,
 * (2)     dLoss / dDenominator = dLoss / dOut * dOut / dDenominator
 * (3)                          = dOut  * - numerator / denominator ** 2
 * (4)                          = dOut * -1 * out / denominator.
 *
 * With numpy broadcasting, (4) gets reduced to the shape of denominator.
 *
 * Eqn. (4) is the gradient of the denominator, the gradient of the numerator
 * is (5)     dLoss / dNumerator = (dLoss / dOut) / numerator.
 *
 * */
class DivAutodiffer {

public:
  // Note that the numerator is not required to compute the gradients. This
  // means that the inplace version of division can be differentiated.
  static std::vector<InIndex> autodiffRequiredIns() { return {1}; }

  static std::vector<OutIndex> autodiffRequiredOuts() { return {0}; }

  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const OpHelper &op) {

    // Equation (4):
    auto denominator = gIn.input(1);
    auto dOut        = gIn.gradOfOutput(0);
    auto out         = gIn.output(0);
    auto dDenominator =
        OpHelper::constantLike(dOut, -1.) * dOut * out / denominator;

    // The gradient of the numerator:
    auto dNumerator = dOut / denominator;

    return {dNumerator.reduceSum(op.inShape(0)),
            dDenominator.reduceSum(op.inShape(1))};
  }
};

/**
 * Differentiation through the unary operation which negates all values.
 * */
class NegAutodiffer {
public:
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const Args &...) {
    // Negate the gradient of the output to get the gradient of the input.
    return {gIn.gradOfOutput(0).neg()};
  }
};

/**
 * Differentiation through f(x) = 1/x. That is, the unary operation which
 * inverts all values of a tensor.
 * */
class InvAutodiffer {
public:
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  /**
   * No inputs of the forward op are required, but the output is:
   * */
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {0}; }

  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const Args &...) {
    // f(x)    =   1/x                   (1)
    // df/dx   =  -1/x^2                 (2)
    //         =  -1 * (1/x) * (1/x)     (3)
    //         =  -1 * f(x)^2            (4) the formulation used below.
    auto out  = gIn.output(0);
    auto dOut = gIn.gradOfOutput(0);
    return {dOut.neg().mul(out.pow(2))};
  }
};

/**
 * Differentiate through y = e^x (where e is the transcendental 2.71828...).
 * */
class ExpAutodiffer {
public:
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {0}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  template <typename Tensor, typename OptionalTensor, typename... Args>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const Args &...) {
    // dIn = out * dOut.
    return {gIn.gradOfOutput(0) * gIn.output(0)};
  }
};

/**
 * Differentiate through the square root operator.
 * */
class SqrtAutodiffer {
public:
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {0}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const OpHelper &op) {

    // dIn = dOut * 1/2 * 1 / sqrt(In)
    //     = dOut * 1/2 / out.
    auto gradOut = gIn.gradOfOutput(0);
    auto half    = op.constantLike(gradOut, 0.5);
    return {half * gradOut / gIn.output(0)};
  }
};

/**
 * Gradient of the power operator.
 *
 * Compute the gradient of the inputs #base and #expo in:
 *
 *   out = base^expo
 *       = exp(log(base) * expo).
 *
 *   dLoss / dBase  = dLoss / dOut * dOut / dBase
 *                  = dLoss / dOut * (expo) * base^(expo-1)
 *
 *   dLoss / dExpo  = dLoss / dOut * dOut / dExpo
 *                  = dLoss / dOut * log(base) * base^expo.
 **/

class PowAutodiffer {

public:
  static bool gradientPropagates(OutIndex, InIndex) { return true; }
  static std::vector<InIndex> autodiffRequiredIns() { return {0, 1}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {0}; }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const OpHelper &) {

    auto outGrad  = gIn.gradOfOutput(0);
    auto base     = gIn.input(0);
    auto exponent = gIn.input(1);

    // The output should be available, if autodiffRequiredOuts of this class
    // is used.
    auto out = gIn.hasOutput(0) ? gIn.output(0) : base.pow(exponent);

    auto dBase = outGrad * exponent *
                 base.pow(exponent - OpHelper::constantLike(outGrad, 1.0));
    auto dExponent = outGrad * base.log() * out;

    return {dBase.reduceSum(base.shape()),
            dExponent.reduceSum(exponent.shape())};
  }
};

/**
 * Differentation of binary ops, 'max' and 'min'.
 *
 * Consider the case of the 'max' operetion, where in0 and in1 are
 * numpy-broadcastable with each other,
 *
 *    out = max(in0, in1).
 *
 * Assume for now that in0 and in1 have the same shape, and that in0 !=
 * in1 for all elements, then
 *
 *    dIn1 = (in1 == out)*dOut, and
 *    dIn0 = (in0 == out)*dOut.
 *
 * If only in1 is available during backpropagation, as is true if the forward
 * operation is done inplace on in0, then the above equations can be expressed
 * as:
 *
 *    mask1 = (in1 == out0)      (1)
 *    dIn1  = mask1 * dOut       (2)
 *    dIn0  = (1 - mask1)*dOut   (3)
 *
 * If the inputs do not have the same shpae, then a sum-reduction down to the
 * input shape is required.
 *
 * For the case of elements where in0 == in1, the function is technically not
 * differentiable, but we do not modify our implementation. Our implemetation
 * has the advantage that for if
 *
 *    out = max(A, A)           (4)
 *
 * then,
 *
 *    dA = dOut.                (5)
 * */
class ExtremumAutodiffer {

public:
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  /**
   * Differentiation requires the input at index 1, and the output.
   * */
  static std::vector<InIndex> autodiffRequiredIns() { return {1}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {0}; }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn,
                const OpHelper &helper) {

    auto outGrad = gIn.gradOfOutput(0);
    auto in1     = gIn.input(1);
    auto out     = gIn.output(0);

    const auto dataType = helper.outDType(0);

    auto mask1 = in1.equalTo(out).to(dataType);
    auto mask0 = mask1.constant(1) - mask1;

    return {(outGrad * mask0).reduceSum(helper.inShape(0)),
            (outGrad * mask1).reduceSum(helper.inShape(1))};
  }
};

/**
 * Differentiate through the subtraction operator. This is like AddAutodiffer,
 * but with the gradient of the second input multiplied by -1.
 * */
class SubAutodiffer {

public:
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return true; }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const OpHelper &op) {
    auto gOut = gIn.gradOfOutput(0);
    return {gOut.reduceSum(op.inShape(0)),
            OpHelper::constantLike(gOut, -1) * gOut.reduceSum(op.inShape(1))};
  }
};

/**
 * Propagate zero to all inputs.
 * */
class ZeroPropagationAutodiffer {

public:
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }
  static bool gradientPropagates(OutIndex, InIndex) { return false; }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &, const OpHelper &) {
    throw poprithms::error::error("autodiff::automatic",
                                  "should not be called");
  }
};

/**
 * Differentiation through an op which copies the input at one index
 * (SourceOfCopy) to the input at one or several others.
 * */
template <int SourceOfCopy> class CopyAutodiffer {
public:
  static std::vector<InIndex> autodiffRequiredIns() { return {}; }
  static std::vector<OutIndex> autodiffRequiredOuts() { return {}; }

  static bool gradientPropagates(OutIndex, InIndex i) {

    // For an input which is the copy destination, a zero gradient is always
    // progatated back (i.e. not propagated by the definition of this
    // function). This is because the value of input at a destination index
    // does not effect the value of the output (which is just a copy of
    // Source).
    return i == SourceOfCopy;
  }

  template <typename Tensor, typename OptionalTensor, typename OpHelper>
  static typename std::vector<OptionalTensor>
  backpropagate(const OpIn<Tensor, OptionalTensor> &gIn, const OpHelper &op) {
    std::vector<OptionalTensor> gradIns(op.nInTensors());
    gradIns[SourceOfCopy] =
        gIn.gradOfOutput(0).reduceSum(op.inShape(InIndex(SourceOfCopy)));
    return gradIns;
  }
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
