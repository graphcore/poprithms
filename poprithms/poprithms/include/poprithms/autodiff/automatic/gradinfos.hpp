// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_GRADINFOS_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_GRADINFOS_HPP

#include <map>
#include <vector>

#include <poprithms/autodiff/core/gradinfo.hpp>
#include <poprithms/autodiff/core/summary.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::autodiff::core::GradInfo;
using poprithms::autodiff::core::Summary;
using poprithms::autodiff::guide::Objective;
using poprithms::autodiff::guide::Traversals;

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::program::callstack::CalleeIndex;

/**
 * A container of core::GradInfo objects, with the extension for callee-caller
 * relationships between graphs.
 * */
class GradInfos {

public:
  /**
   * Register that the sub-graph #gradId is the gradient sub-graph with
   * objective/summary defined by #gradInfo.
   * */
  void insert(SubGraphId gradId, const GradInfo &gradInfo);

  /**
   * Get the gradient information of the gradient sub-graph #gradId.
   * */
  const GradInfo &at(SubGraphId gradId) const;

  /**
   * All the gradient graphs created for the objective #objective. Recall that
   * in this project there can be multiple gradient graphs of any sub-graph.
   * It is even possible to have multiple gradient graphs for a single
   * sub-graph and a single objective (although this might be a strange this
   * for user to do).
   * */
  SubGraphIds gradGraphsCreatedFor(const Objective &objective) const;

  /**
   * Return true if there is a gradient sub-graph registered for the #ci'th
   * callee of the op #opId.
   * */
  bool hasGrad(OpId opId, CalleeIndex ci) const;

  /**
   * Set the gradient of the #ci'th callee of the op #opIf to #sgId.
   * */
  void setGrad(OpId opId, CalleeIndex ci, SubGraphId sgId);

  SubGraphId grad(OpId opId, CalleeIndex ci) const;

private:
  // A map from a gradient sub-graph to a single gradient info (what sub-graph
  // is it the gradient of, and what was the objective (targets in sub-graph,
  // checkpoints, etc)).
  std::map<SubGraphId, GradInfo> gradInfos_;

  // Map from an objective to all of the gradient graphs created for it.
  std::map<Objective, SubGraphIds> gradsForObjective_;

  // The gradients of the callees of an op (keys will be ops with callees:
  // switch, if, repeat, etc.).
  std::map<OpId, std::vector<SubGraphId>> gradsForCallees;
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
