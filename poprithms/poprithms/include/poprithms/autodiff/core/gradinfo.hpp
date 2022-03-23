// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_CORE_GRADINFO_HPP
#define POPRITHMS_AUTODIFF_CORE_GRADINFO_HPP

#include <array>
#include <map>
#include <set>
#include <vector>

#include <poprithms/autodiff/core/summary.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

using poprithms::autodiff::core::Summary;
using poprithms::autodiff::guide::Objective;

using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;

/**
 * Descriptor for a gradient graph, and its relation to a non-gradient
 * graph. This is a utility class for connecting gradient and non-gradient
 * tensors.
 *
 * */
struct GradInfo {

public:
  /**
   * \param nonGradGraph The undifferentiated (non-gradient) graph. All
   *                     tensors in this graph are non-gradient tensors.
   *
   * \param gradGraph The gradient graph of #nonGradGraph. Some tensors in
   *                  this graph are gradient tensors, some are non-gradient
   *                  tensors.
   *
   * \param objective The objective used to create #gradGraph from
   *                  #nonGradGraph. This describes the tensors in
   *                  #nonGradGraph which are (1) the targets of
   *                  differentiation, (2) the checkpoints, and (3) the
   *                  tensors which have gradients provided for them in the
   *                  gradient graph.
   *
   * \param summary The summary of the tensors in the gradient graph
   *                corresponding to the tensors in #objective.
   *
   * */
  GradInfo(SubGraphId nonGradGraph,
           SubGraphId gradGraph,
           const Objective &objective,
           const Summary &summary);

  SubGraphId gradSubGraphId() const { return gradSubGraphId_; }

  SubGraphId nonGradSubGraphId() const { return nonGradSubGraphId_; }

  /**
   * Checkpoint tensors are computed in the non-gradient graph, then copied to
   * the gradient graph, where they are used to compute gradient tensors. This
   * method returns the location of the copy in the gradient graph for a
   * checkpoint tensor #inNonGradGraph in the non-gradient graph.
   *
   * \param inNonGradGraph A non-gradient tensor in the non-gradient graph.
   *
   * \return The non-gradient tensor in the gradient graph, to which the
   *         non-gradient tensor #inNonGradGraph is copied.
   * */
  TensorId checkpointInGradGraph(const TensorId &inNonGradGraph) const;

  /**
   * The inverse of the method #checkpointInGradGraph.
   * */
  TensorId checkpointInNonGradGraph(const TensorId &inGradGraph) const;

  struct CheckpointPair {
    TensorId inNonGradGraph;
    TensorId inGradGraph;
  };
  using CheckpointPairs = std::vector<CheckpointPair>;

  /**
   * \return all checkpoint pairs, created by 'zipping' tensors in then
   *         objective and the summary together.
   * */
  CheckpointPairs checkpointPairs() const;

  /**
   * \param inNonGradGraph A non-gradient tensor in the non-gradient graph.
   *
   * \return The gradient tensor in the gradient graph, to which the gradient
   *         of #inNonGradGraph is copied.
   * */
  TensorId gradInputInGradGraph(const TensorId &inNonGradGraph) const;

  /**
   * The inverse of the method #gradInputInGradGraph. This returns a
   * non-gradient tensor in the non-gradient graph.
   * */
  TensorId gradInputInNonGradGraph(const TensorId &gradInGradGraph) const;

  struct GradInPair {
    TensorId nonGradInNonGradGraph;
    TensorId gradInGradGraph;
  };
  using GradInPairs = std::vector<GradInPair>;

  /**
   * \return all gradient input pairs, created by 'zipping' tensors in
   *         objective and summary together.
   * */
  GradInPairs gradInPairs() const;

  /**
   * \param inNonGradGraph A non-gradient tensor in the non-gradient graph,
   *                       for which a gradient must be computed in the
   *                       gradient graph.
   *
   * \return A gradient tensor in the gradient graph. It is the gradient of
   *         #inNonGradGraph
   * */
  TensorId targetGradInGradGraph(const TensorId &inNonGradGraph) const;

  /**
   * The inverse of the method #targetGradInGradGraph.
   * */
  TensorId targetInNonGradGraph(const TensorId &inGradGraph) const;

  struct TargetAndGradPair {
    TensorId nonGradInNonGradGraph;
    TensorId gradInGradGraph;
  };
  using TargetAndGradPairs = std::vector<TargetAndGradPair>;

  /**
   * \return All gradients of targets in the gradient graph.
   * */
  TensorIds targetGradsInGradGraph() const { return summary_.targetGrads(); }

  /**
   * Construct a GradInfo from 'zipped' pairs, rather than from an objective
   * and summary.
   * */
  static GradInfo outOfGraph(SubGraphId nonGradSubGraphId,
                             SubGraphId gradSubGraphId,
                             const GradInPairs &grads,
                             const CheckpointPairs &checkpoints,
                             const TargetAndGradPairs &targets);

  const Summary &summary() const { return summary_; }

  const Objective &objective() const { return objective_; }

private:
  SubGraphId nonGradSubGraphId_;
  SubGraphId gradSubGraphId_;
  Objective objective_;
  Summary summary_;
};

} // namespace core
} // namespace autodiff
} // namespace poprithms

#endif
