// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_GUIDE_OBJECTIVE_HPP
#define POPRITHMS_AUTODIFF_GUIDE_OBJECTIVE_HPP

#include <ostream>
#include <string>

#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

/**
 * A high-level descriptor of the required outcome of differentiating a graph.
 *
 * What is overall objective of the differentiation?
 *
 * 1) Which tensors have gradients provided for? These are the starting points
 * of the back-propagation. The vanilla case is the loss scalar tensor, whose
 * gradient is the scalar Tensor with value '1'. But these tensors can be any
 * tensors in the graph.
 *
 * 2) Which tensors are checkpoints? If there is to no recomputation, then all
 * the tensors required for back-propagation must be checkpoints. Any tensors
 * which are required for back-propagation and are not checkpoints, will be
 * recomputed during back-propagation. Certain tensors cannot be recomputed,
 * such as graph inputs, and these must always be in the set of checkpointed
 * tensors.
 *
 * 3) Which tensors must have there gradients computed? The vanilla case for
 * this is 'all weight tensors', but this set of tensors can be anything.
 *
 * 4) Finally, should the graph be differentiated "in situ", whereby the graph
 * is extended? Or should the gradient operations be contained in a separate
 * graph? There are separate factory constructors for these 2 cases. The "in
 * situ" constructor requires, for each tensor in (1) above, a corresponding
 * gradient tensor. For the vanilla loss case, this will be a scalar tensor
 * with value 1.
 * */

struct Objective {

  static Objective outOfGraph(const TensorIds &gradsProvidedFor,
                              const TensorIds &checkpoints,
                              const TensorIds &targets);

  static Objective inGraph(const TensorIds &gradsProvidedFor,
                           const TensorIds &checkpoints,
                           const TensorIds &targets,
                           const TensorIds &gradsProvided);

  /**
   * The tensors with input gradients. In PyTorch terms, these are the
   * tensors which the method 'backward' is called on.
   * */
  const TensorIds &gradsProvidedFor() const { return gradsProvidedFor_; }
  TensorId gradProvidedFor(uint64_t i) const {
    return gradsProvidedFor_.at(i);
  }
  bool hasGradProvided(const TensorId &) const;
  uint64_t nInGrads() const { return gradsProvidedFor_.size(); }

  /**
   * The tensors whose values will be available during backpropagation.
   * Any tensors which are needed but are not available will need to be
   * recomputed.
   * */
  const TensorIds &checkpoints() const { return checkpoints_; }
  TensorId checkpoint(uint64_t i) const { return checkpoints_.at(i); }
  uint64_t nCheckpoints() const { return checkpoints_.size(); }
  bool isCheckpoint(const TensorId &inNonGrad) const;

  /**
   * The tensors which the graph differentiation must ultimately
   * create gradients for. In PyTorch terms, this is the set of all tensors
   * which have 'requires_grad=True'
   * */
  const TensorIds &targets() const { return targets_; }
  TensorId target(uint64_t i) const { return targets_.at(i); }
  uint64_t nTargets() const { return targets_.size(); }
  bool isTarget(const TensorId &) const;

  const TensorIds &gradsProvided() const;

  bool isInGraph() const { return inGraph_ == InGraph::Yes; }

  void append(std::ostream &) const;
  std::string str() const;

  // Note that if the order of targets in the objective is permuted, then this
  // comparison operator return false.
  bool operator==(const Objective &r) const { return t() == r.t(); }
  bool operator<(const Objective &r) const { return t() < r.t(); }

  TensorIds allTensorIds() const;

private:
  TensorIds gradsProvidedFor_;
  TensorIds checkpoints_;
  TensorIds targets_;

  enum class InGraph { No, Yes } inGraph_;
  TensorIds gradsProvided_;

  std::tuple<TensorIds, TensorIds, TensorIds, InGraph, TensorIds> t() const {
    return {
        gradsProvidedFor_, checkpoints_, targets_, inGraph_, gradsProvided_};
  }

  Objective(const TensorIds &gradsProvidedFor,
            const TensorIds &checkpoints,
            const TensorIds &targets,
            InGraph,
            const TensorIds &gradsProvided);
};

std::ostream &operator<<(std::ostream &, const Objective &);

} // namespace guide
} // namespace autodiff
} // namespace poprithms

#endif
