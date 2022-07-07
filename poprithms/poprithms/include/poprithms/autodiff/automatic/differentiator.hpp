// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_DIFFERENTIATOR_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_DIFFERENTIATOR_HPP

#include <memory>
#include <string>
#include <vector>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/callstack/carriedtensorid.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using autodiff::core::GradInfo;
using autodiff::guide::Objective;
using common::multiout::InIndex;
using common::multiout::InIndices;
using common::multiout::OpId;
using common::multiout::OptionalTensorIds;
using common::multiout::OutIndex;
using common::multiout::OutIndices;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;

/**
 * A class for performing high-level, global, graph differentation of first
 * and (proof of concept) second order. This builds on-top of the foundation
 * classes in the namespace autodiff::guide and autodiff::core, and requires
 * additional virtual methods to be implemented to generate gradients. For
 * example, the virtual methods must provide information about tensor shapes
 * and types, which are not needed at the abstraction levels of
 * autodiff::guide and autodiff::core.
 * */
class Differentiator {

  // This object records the relationship between tensors and their gradients.
  // As a comparison, in PyTorch, one obtains the gradient of a tensor #t by
  // calling #t.grad. This class essentially performs that role. The design
  // pattern that is used here, however, ensures that automatic
  // differentiation is independent of the application's underlying graph
  // intermediate representation.
  autodiff::automatic::GradInfos gradInfos_;

  // Virtual classes with 'getters and setters' which define how to actually
  // generate ops in the user's graph.
  const IAutomaticQuerier &querier;
  IAutomaticMutator &mutator;

public:
  virtual ~Differentiator();
  Differentiator(const IAutomaticQuerier &aq, IAutomaticMutator &am)
      : querier(aq), mutator(am) {}

  /**
   * Create gradients of the loss with respect to each of the tensors in
   * #vars. #loss should be a scalar tensor, and in the same graph as all the
   * tensors in #vars. The new gradients will be in the same graph as the
   * #loss and #vars tensors.
   * */
  TensorIds backward(const TensorId &loss, const TensorIds &vars);

  /**
   * Create gradients of some scalar #v with respect to each of the tensors in
   * #targets. Gradients of #v are provided as inputs for
   * #gradsProvidedFor, with the actual gradients being the tensors
   * #gradsProvided. All tensors in #checkpoints can be used during
   * backpropagation. If there is a tensor which is required that is not in
   * #checkpoints, it will be recomputed.
   *
   * All tensors, including the returned gradient tensors, are in the same
   * graph.
   * */
  TensorIds backwardInGraph(const TensorIds &gradsProvidedFor,
                            const TensorIds &checkpoints,
                            const TensorIds &targets,
                            const TensorIds &gradsProvided);

  /**
   * Create a gradient graph which computes the gradients of tensors in
   * #targets, provided with input gradients for #gradsProvidedFor.
   * */
  SubGraphId backwardOutOfGraph(const TensorIds &gradsProvidedFor,
                                const TensorIds &checkpoints,
                                const TensorIds &targets);

  /**
   * This method calls directly into the other #backwardOutOfGraph.
   * */
  SubGraphId backwardOutOfGraph(const Objective &o);

  /**
   * Set the gradient graph of the callee #ci of op #op to #grad.
   * */
  void setGrad(OpId op, CalleeIndex ci, SubGraphId grad);

  /**
   * The minimal set of tensors required to compute the gradients of #targets
   * without requiring any recomputation.
   * */
  TensorIds
  minimalNonRecomputationCheckpoints(const TensorIds &gradsProvidedFor,
                                     const TensorIds &targets);

  virtual const autodiff::guide::GraphInfo &graphInfo() const = 0;

  virtual std::unique_ptr<autodiff::core::GraphMutator>
      graphMutator(SubGraphId) const = 0;

  virtual std::unique_ptr<Differentiator> cloneWithoutGradInfo() const = 0;

  const GradInfo &gradInfo(SubGraphId sgId) const {
    return gradInfos_.at(sgId);
  }

  const autodiff::automatic::GradInfos &gradInfos() const {
    return gradInfos_;
  }

  void insertGradInfo(const GradInfo &);

  /**
   * It is possible to create graphs which compute Jacobian and
   * Hessians in term of the first-order gradient methods above. We provide 2
   * examples here, although these are just to demonstrate that it is possible
   * (the implementations are a few tens of lines).
   *
   * Hessian tensors are often prohibitively large, and expensive to
   * compute. In practise it is common to implicitly compute projections
   * with them. See for example Hessian free methods (Deep learning via
   * Hessian-free optimization ICML 2009) and the PyHessian paper, where the
   * principle eigenvector of the Hessian is computed by the power method, but
   * the Hessian itself is not computed.
   *
   * \sa https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf
   *     for Deep learning via Hessian-free optimization.
   *
   *  \sa https://arxiv.org/abs/1912.07145 for PyHessian.
   *
   * Consider
   *   L = f(X) where f : R^m -> R^1.
   *
   * The Hessian is a function
   *   h : R^(m) -> R^(m * m).
   *
   * which computes d/dX (dL/dX). That is, the second derivative of L w.r.t.
   * X.
   *
   * Given a vector v in R^(m), The Hessian projects v to a new vector in
   * R^(m). That is what the following method does, where #v is called a
   * 'projection' tensor.
   *
   * The function name #hvp is chosen to agree with PyTorch function naming.
   * */
  struct HessianProjections {

    // For each target, these are the products between the Hessian of the
    // target and the projection vector.
    TensorIds projectedTargets;

    // The projection tensors.
    TensorIds projections;
  };
  HessianProjections hvp(const TensorId &loss, const TensorIds &targets);

  /**
   * The complete Hessian tensor (which is seldom required in practise) is
   * constructed by computing the projection for all one-hot vectors for the
   * indices of the #target tensor.
   *
   * This corresponds to the PyTorch Hessian method with vectorize=False.
   * */
  struct CompleteHessian {
    SubGraphId hessianGraph;
    TensorId targetInHessianGraph;
    TensorId hessian;
  };
  CompleteHessian completeHessian(const TensorId &loss,
                                  const TensorId &target);

private:
  void createMissingGradGraphs(const Objective &);

  autodiff::core::Summary getSummary(const Objective &, SubGraphId bwd);

  void verifyInForwardGraphOf(SubGraphId bwd, const TensorId &inFwd) const;
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
