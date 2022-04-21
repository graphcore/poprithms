// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PRUNE_PRUNE_HPP
#define POPRITHMS_PROGRAM_PRUNE_PRUNE_HPP

#include <ostream>
#include <string>
#include <vector>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/querier.hpp>
#include <poprithms/program/callstack/stacktensorid.hpp>

namespace poprithms {
namespace program {
namespace prune {

using poprithms::common::multiout::ConsumptionId;
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;
using poprithms::program::callstack::CallStack;
using poprithms::program::callstack::StackTensorId;
using poprithms::program::callstack::StackTensorIds;

class Mutator {
public:
  /**
   * Remove the inputs at input indices #ins from op #opId. This will only be
   * called (by Pruner::prune) on ops with callee sub-graphs. That is, only
   * inputs which are copies to callees will ever be removed, and only if they
   * are not on a path to a required tensor. \sa the Pruner class.
   * */
  virtual void removeInputs(OpId opId, const InIndices &ins) = 0;

  /**
   * Remove the outputs at output indices #outs from op #opId. This will only
   * be called (by Pruner::prune) on ops with callee sub-graphs.
   * */
  virtual void removeOutputs(OpId opId, const OutIndices &outs) = 0;

  /**
   * Remove the op #opId. The will only be called (by Pruner::prune) on ops
   * whose outputs have no consumers.
   * */
  virtual void removeOp(OpId opId, const std::string &ctxt) = 0;
};

class Pruner {
public:
  /**
   * \param querier defines the DAG structure of the graph, and the callee
   *                hierarchy.
   *
   * \param callables these are all entry-point graphs. These can be thought
   *                  of as all sub-graphs which a user might call/execute.
   *
   * \param backSources this is a set of tensors which must not be pruned.
   *
   * If a user might call any sub-graph in #callables, which tensors must be
   * retained to ensure that the tensors in #backSources can be computed?
   * */
  static TensorIds unpruneable(const callstack::Querier &querier,
                               const SubGraphIds &callables,
                               const TensorIds &backSources);

  /**
   * Using #querier, #callables, and #backSources, obtain a set of unpruneable
   * tensors (\sa the method above). Then,
   *
   * 1) remove all ops which do not have any unpruneable outputs.
   * 2) remove all copies into / out of callees where the destination of the
   *    copy is not unpruneable.
   * */
  static void prune(const callstack::Querier &querier,
                    Mutator &mutator,
                    const SubGraphIds &callables,
                    const TensorIds &backSources);
};

} // namespace prune
} // namespace program
} // namespace poprithms

#endif
