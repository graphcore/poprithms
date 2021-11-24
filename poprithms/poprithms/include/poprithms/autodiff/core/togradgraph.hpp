// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRRITHMS_AUTODIFF_CORE_TOGRADGRAPH_HPP
#define POPRRITHMS_AUTODIFF_CORE_TOGRADGRAPH_HPP

#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

/**
 * Base class for an object which can map from tensors in an undifferentiated
 * graph, to the corresponding gradient and non-gradient tensors in its
 * derivate graph.
 * */
class ToGradGraph {
public:
  virtual ~ToGradGraph() = default;

  /**
   * \return a vector of the same length as input #inNonGrad. At index i the
   *         returned vector has value:
   *         1) the gradient of inNonGrad[i] if inNonGrad[i] is a tensor in
   *            the non-gradient graph, and
   *         2) none, otherwise.
   * */
  virtual OptionalTensorIds
  optionalGrads(const TensorIds &inNonGrad) const = 0;

  /**
   * \return a vector of the same length as input #inNonGrad. At index i the
   *         returned vector has value:
   *         1) the non-gradient tensor in the gradient graph corresponding
   *            to inNonGrad[i] (either recomputed, or checkpointed) if
   *            inNonGrad[i] is a tensor in the non-gradient graph, or
   *         2) none, otherwise.
   * */
  virtual OptionalTensorIds
  optionalNonGrads(const TensorIds &inNonGrad) const = 0;

  /**
   * The gradient of the tensor #inNonGrad. If #inNonGrad is not a tensor in
   * the non-gradient graph, then an error is thrown.
   * */
  virtual TensorId getGrad(const TensorId &inNonGrad) const = 0;

  /**
   * The non-gradient (either recomputed or checkpointed) of the tensor
   * #inNonGrad. If #inNonGrad is not a tensor in the non-gradient graph, then
   * an error is thrown.
   * */
  virtual TensorId getNonGrad(const TensorId &inNonGrad) const = 0;
};

} // namespace core
} // namespace autodiff
} // namespace poprithms

#endif
