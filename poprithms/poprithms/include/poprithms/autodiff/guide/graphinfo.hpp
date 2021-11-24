// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_AUTODIFF_GUIDE_GRAPHINFO_HPP
#define POPRITHMS_AUTODIFF_GUIDE_GRAPHINFO_HPP

#include <set>

#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

class ToGradGraph;

/**
 * Provides basic information about the graph to be differentiated.
 * */
class GraphInfo {

public:
  virtual ~GraphInfo() = default;

  /**
   * Can a gradient be propagated through the OpTraversal #ot? Specifically,
   * if there is a non-zero gradient at the output index of #ot, might there
   * be a resulting non-zero gradient at the input index of #ot?
   * */
  virtual bool gradientPropagates(const OpTraversal &ot) const = 0;

  /**
   * The Op #opId requires zero, one or several activations to backpropagate
   * the gradients of its outputs to its inputs. The activations may be inputs
   * or outputs, and they may be optional. For example, when backpropagating
   * through, Out = relu(In),
   *
   * it is sufficient to have either #Out or #In, as,
   *    dLoss/dIn = dLoss/dOut * (Out > 0)
   *              = dLoss/dOut * (In > 0).
   *
   * An implementation of this method must ensure that either #Out or #In is
   * inserted into #required.
   *
   * As a second example, consider Out = matmul(A, B). In this case,
   *   dLoss/dA = reduceSum(matmul(dLoss/dOut, B.T)), and
   *   dLoss/dB = reduceSum(matmul(A.T, dLoss/dOut)), and so this method must
   * ensure that both A and B are in #required.
   *
   * As a third example, consider Out = A + B. In this case,
   *    dLoss/dA = reduceSum(dLoss/dOut), and
   *    dLoss/dB = reduceSum(dLoss/dOut).
   * As there is no appearance of A or B on the right hand side of these
   * equations, this method does not need to insert any tensors into #required
   * for an addition op.
   * */

  virtual void
  extendAutodiffRequiredTensors(OpId opId,
                                std::set<TensorId> &required) const = 0;

  /**
   * \return the OpIds in #opIds, sorted into a valid topological order.
   * */
  virtual OpIds subSchedule(const std::set<OpId> &opIds) const = 0;

  /**
   * Append information about #opId to the stream #ost. This is used for
   * logging and error messages.
   * */
  virtual void appendOpInfo(std::ostream &ost, OpId opId) const = 0;

  /**
   * \return the input tensors of #opId.
   * */
  virtual TensorIds inTensorIds(OpId opId) const = 0;

  /**
   * \return the input at index #i of op #opId.
   * */
  virtual TensorId inTensorId(OpId opId, InIndex i) const = 0;

  /**
   * \return the number of input tensors of op #opId.
   * */
  virtual uint64_t nInTensors(OpId opId) const = 0;

  /**
   * \return the number of output tensors of op #opId
   **/
  virtual uint64_t nOutTensors(OpId) const = 0;

  /**
   * \return the consumers of the tensor #tId.
   * */
  virtual ConsumptionIds consumptionIds(const TensorId &tId) const = 0;

  /**
   * For certain ops, such as a 'variable initializer', it might not make
   * sense to rerun them to recompute their outputs. For improved debugging,
   * this method should throw an error for such ops.
   * */
  virtual void assertCanBeRerun(OpId) const = 0;

  /**
   * Certain tensors, such an tensors of integral types, it might never make
   * sense to have a corresponding gradient tensor. For improved debugging,
   * this method should throw an error for such ops.
   * */
  virtual void assertCanHaveGrad(const TensorId &) const = 0;

  /**
   * Certain combinations of 'targets' and 'gradsProvidedFor' in the Objective
   * might suggest a user error. An example is when not all tensors belong to
   * the same graph. For improved debugging, this method should throw an error
   * for such combinations.
   * */
  virtual void assertValidPaths(const TensorIds &targets,
                                const TensorIds &gradsProvidedFor) const = 0;

  /** Does it propagate back along any path? */
  bool gradientPropagates(const TensorId &id) const;

  TensorId inTensorId(const OpTraversal &ot) const;
  TensorIds outTensorIds(OpId) const;
};
} // namespace guide

} // namespace autodiff
} // namespace poprithms

#endif
