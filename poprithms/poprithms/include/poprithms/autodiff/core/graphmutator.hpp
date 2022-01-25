// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRRITHMS_AUTODIFF_CORE_GRAPHMUTATOR_HPP
#define POPRRITHMS_AUTODIFF_CORE_GRAPHMUTATOR_HPP

#include <poprithms/autodiff/core/togradgraph.hpp>
#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

/**
 * The class for creating new tensors in the gradient graph during
 * backpropagation. This class 'directs' the algorithm in
 * Autodiff::backpropagte as to how exactly new tensors are created. This
 * class 'does calculus'.
 * */
class GraphMutator {
public:
  virtual ~GraphMutator() = default;

  /**
   * Create a zero tensor which is "like" #like (shape, type, location, etc.).
   * The tensor can be constant, and it can contain self-aliases.
   * */
  virtual TensorId createZero(const TensorId &like) = 0;

  /**
   * Create a variable tensor which is "like" #like  (shape, type, location,
   * etc.). The tensor should not contain self-aliases, as it will be the
   * destination of a copy of a tensor in the forward graph.
   * */
  virtual TensorId createVariable(const TensorId &like) = 0;

  /**
   * Create a clone of #opId in the gradient graph, which has inputs #ins.
   * */
  virtual OpId clone(OpId opId, const TensorIds &ins) = 0;

  /**
   * Sum a (non-empty) set of tensors. This an be implemented using a single
   * sum op, or as a tree of adds. The output tensor may alias an input
   * tensor.
   *
   * \param toSum The gradient tensors to sum.
   *
   * When called by the Autodiff class, #toSum will
   * (1) always be at least one tensor.
   * (2) be ordered from first created (earliest in the backpropagation) to
   *     last created.
   * */
  virtual TensorId sum(const TensorIds &toSum) = 0;

  // TODO(T50868) the user of this project (popart) might prefer the gradients
  // to be accumulated in a sum op, as opposed to in a tree of add ops.

  /**
   * Set the name of #opId to #n. This is used for logging and error messages.
   * */
  virtual void setName(OpId, const std::string &) = 0;

  /**
   * Generate gradients of the inputs to the forward op #opId. This is where
   * 'calculus' must be implemented.
   *
   * \param toGradGraph an object to map from tensors in the forward graph, to
   *        tensors in the backwards graph.
   *
   * \return A vector whose length is the number of outputs of #opId. Each
   *         element in the vector is either 'none', if no gradient was
   *         propagated to the tensor, or it is the gradient of the input
   *         tensor.
   *
   * Example. Consider `z = mul(x, y).`, where
   *     dx = reduce(mul(dz, y))
   *     dy = reduce(mul(dz, x)).
   *
   * The implementation of this might look something like
   *
   *     return { toGradGraph.getGrad(z).getNonGrad(y).reduce(..),
   *              toGradGraph.getGrad(z).getNonGrad(x).reduce(..) };
   * */
  virtual OptionalTensorIds getInGrads(OpId,
                                       const ToGradGraph &toGradGraph) = 0;
};

} // namespace core
} // namespace autodiff
} // namespace poprithms

#endif
