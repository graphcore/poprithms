// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PRUNE_CALLSTACK_GRAPHINTERFACE_HPP
#define POPRITHMS_PROGRAM_PRUNE_CALLSTACK_GRAPHINTERFACE_HPP

#include <map>
#include <ostream>
#include <vector>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/stacktensorid.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::ConsumptionId;
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;
using poprithms::program::callstack::CallStack;
using poprithms::program::callstack::StackTensorId;
using poprithms::program::callstack::StackTensorIds;

/**
 * Interface (abstract base class) for a graph with ops with callees.
 *
 * Several (simple to interpret) virtual methods must be overriden by a user
 * of class. There are some utility (non-virtual) methods that then use these
 * virtual methods. The virtual methods describe the ops in a node: what the
 * inputs are, how many outputs there are, what the callees (if any) there are
 * etc. The utility methods which call into these virtual methods are for
 * traversing the ops and generating call stacks.
 *
 * This class make 1 assumption on ops with multiple callees. There is no
 * assumption on the inputs -- each callee can have a different number of
 * inputs -- but each callee is assumed to have the same number of outputs.
 * See the CopyIns and CopyOuts classes for more info.
 * */
class Querier {

public:
  /** Number of outputs of #id */
  virtual uint64_t nOutTensors(OpId id) const = 0;

  /** Number of inputs of #id */
  virtual uint64_t nInTensors(OpId id) const = 0;

  /**
   * The sub-graphs which the op #id calls. For a call op, this will be
   * the single callee graph. For a switch op, this will be all the
   * sub-graphs: one for each of the switch cases. For most 'normal' ops, this
   * will be the empty set.
   * */
  virtual SubGraphIds callees(OpId id) const = 0;

  /** The inputs of op #id */
  virtual TensorIds inTensorIds(OpId id) const = 0;

  /** All ops in all the sub-graphs. */
  virtual OpIds opIds() const = 0;

  /** All of the ops in the sub-graph #sg */
  virtual OpIds opIds(SubGraphId sg) const = 0;

  /**
   * A string summary of the op #id. This is used only for logging/debugging.
   */
  virtual std::string str(OpId id) const = 0;

  /**
   * Recall that a CallEvent contains
   *      (1) an op with callee(s) and
   *      (2) one of the callee graphs, and its index with a calling op.
   * This method returns true if #tId is in the callee
   * graph, and is copied into it (before the callee executes). */
  virtual bool isDstInCallee(const TensorId &tId,
                             const CallEvent &) const = 0;

  /**
   * This method assumes that #inCallee is in the callee graph of #ce. That
   * is, the op in #ce calls into a sub-graph which contains inCallee.
   * It also assumes that before executing the sub-graph, there is copy into
   * #inCallee. If these are conditions, it returns the source of this copy.
   * Otherwise an error is thrown.
   * */
  virtual TensorId srcInCaller(const TensorId &inCallee,
                               const CallEvent &ce) const = 0;

  /**
   * What is the source of the #o'th copy-out of #ce's callee? Note that we
   * make an assumption here about the structure of ops with multiple callees
   * (such as a switch op) -- the assumption is that there are the same number
   * of out-copies from each callee graph.
   * */
  virtual TensorId srcInCallee(const CallEvent &ce, OutIndex o) const = 0;

  /**
   * Which ops have #tId as an input, and which index is #tId an input at?
   * */
  virtual ConsumptionIds consumptionIds(const TensorId &tId) const = 0;

  /**
   * This virtual method is used by #unpruneableMultiGraphBackSource to
   * traverse from tensors in a (call) op's callees to tensors in the call
   * op's graph. See the class #CopyInMap, which can help generate a map for
   * this from a graph.
   *
   * It answers: What are the sources of all the copies into #inCallee? That
   * is, if #inCallee is a tensor in a sub-graph, what are the sources of the
   * copies to it?
   * */
  virtual std::vector<std::pair<CallEvent, InIndex>>
  getCopyInsTo(const TensorId &inCallee) const = 0;

  /**
   * Used by #unpruneableMultiGraphBackSource to traverse from callers in
   * callees. See the class #CopyOutMap which helps generate a map for this
   * from a graph.
   *
   * What are the destinations of copies out of #inCallee's sub-graph from
   * #inCallee?
   * */
  virtual std::vector<std::pair<CallEvent, OutIndex>>
  getCopyOutsFrom(const TensorId &inCallee) const = 0;

public:
  /**
   * The following methods call into the virtual methods defined above.
   * */

  /**
   * Perform a (reverse) depth-first search starting from #tIds. This method
   * does not traverse through copies into or out of callee sub-graphs, the
   * Tensor->Tensor traversal is simply to all of a tensor's creators inputs.
   *
   * The word "Single" in the method name is to say that it does not leave
   * traverse out of a graph.
   * */
  TensorIds onSingleGraphPathTo(const TensorIds &tIds) const;

  /**
   * Performs a (reverse) depth-first search starting from #stIds.
   *
   * Tensor->tensor traversals are defined by:
   *
   *   (1) if a tensor's creator op has no callees, traverse to all inputs of
   *       the creator.
   *
   *   (2) if an tensor's creator op has callees, traverse to the sources in
   *       the callees of the out-copy to the tensor.
   *
   *   (3) if an tensor is in a callee and is a in-copy destination,
   *       traverse to the source of the copy in the calling op's sub-graph.
   * */
  StackTensorIds onMultiGraphPathTo(const StackTensorIds &stIds) const;

  bool hasCallees(OpId opId) const { return !callees(opId).empty(); }

  /**
   * Contiguous output tensors of #id:
   * TensorId(id,0)...TensorId(id,nOutputs-1)
   * */
  TensorIds outTensorIds(OpId id) const;

  /**
   * For a sub-graph sg, enumerate all Tensors in #sg. Do this starting from
   * sub-graphs in #stackBases, and do the same for any sub-graph #sg (with
   * #sg added the CallStack).
   *
   * Note that this method does not traverse through copies into/out of
   * copies, it enumerates all tensors in sub-graphs.
   * */
  StackTensorIds nestedFullStack(const SubGraphIds &stackBases) const;

  /**
   * Obtain StackTensors from #nestedFullStack, and convert it to a map.
   * */
  std::map<TensorId, std::vector<CallStack>>
  nestedFullStackMap(const SubGraphIds &) const;
};

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
