// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_WITHCALLEES_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_WITHCALLEES_HPP

#include <map>

#include <poprithms/common/compute/ihostrunner.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/multiout/skiptraversal.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/program/callstack/stackedio.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::program::callstack::CopyIn;
using poprithms::program::callstack::CopyOuts;
using poprithms::program::callstack::StackedCopyOrder;

/**
 * The base class for ops such as Call, Repeat, Switch, etc -- all the ops
 * which have callee sub-graphs.
 * */
class WithCallees : public Op {

private:
  // The sub-graphs which this op calls. For call/repeat this will just be 1
  // sub-graph id, for switch it will be multiple.
  SubGraphIds callees_;

  // The destinations of the copies into the callees.
  CalleeTensorIds inDsts_;

  // The sources of the copies out of the callees.
  CopyOuts outs_;

public:
  /**
   * \param opState The input ids, input types, output types, etc.
   *
   * \param callees The sug-graph(s) to call. There must be at least one
   *                sub-graph (this vector cannot be empty, ops with no
   *                callees can inherit from WithoutCallees).
   *
   * \param inDsts The input copy destinations.
   *
   * \param outs The sources of the copies out of the callees, back into the
   *             calling scope.
   *
   * Recursive calls to sub-graphs are not supported. For example an op cannot
   * have its own sub-graph as one of its callee.
   *
   * This op has some inputs which are copied to sub-graphs, and some which
   * are not. All the copy inputs must appear at input indices lower than the
   * non-copy inputs.
   *
   * Currently this op assumes that all outputs are copied from sub-graphs.
   * */
  WithCallees(const State &opState,
              const SubGraphIds &callees,
              const CalleeTensorIds &inDsts,
              const CopyOuts &outs);

  /**
   * This op has some inputs which are copied to callee sub-graphs, and might
   * have some inputs which are not. This method returns the indices of inputs
   * which are not copied to callee sub-graphs. An example of such an input
   * would be the conditional tensor of a switch op.
   *
   * This class assumes that all of these indices appear contiguously after
   * all indices of copy inputs.
   * */
  InIndices nonCopyToCalleeIndices() const;

  /**
   * All of the inputs which are copied to callee sub-graphs. This is all
   * input indices not in #nonCopyToCalleeIndices.
   *
   * This class assumes that all of these indices appear contiguously before
   * all indices of non-copy inputs.
   * */
  InIndices calleeCopyInIndices() const;

  /**
   * All the input copies are at input indices lower than the non-copy inputs.
   * */
  bool isCopyToCalleeInIndex(InIndex i) const final {
    return i.get() < nInputsCopiedToCallees();
  }

  /**
   * The destinations in the callee sub-graphs of the input copies. The size
   * of the returned vector is the same as that returned by
   * #calleeCopyInIndices, and there is a 1:1 correspondence.
   * */
  const CalleeTensorIds &inDsts() const { return inDsts_; }

  /**
   * The destinations in the callee sub-graphs. This is the same as the vector
   * returned by #inDsts, but with the callee sub-graph indices removed.
   * */
  TensorIds inTensorIdDsts() const;

  /**
   * The destination of the #i'th input copy.
   *
   * \param i An input index, must be less than #nInputsCopiedToCallees.
   * */
  CalleeTensorId dstInCallee(InIndex i) const final;

  /**
   * The number of inputs which are copied to callee sub-graphs.
   * */
  uint64_t nInputsCopiedToCallees() const final { return inDsts_.size(); }
  uint64_t nInCopies() const { return nInputsCopiedToCallees(); }

  /**
   * The copy destinations in callee sub-graphs of the inputs at indices
   * #inds.
   * */
  TensorIds inDsts(const InIndices &inds) const;

  /**
   * The sources of all the copies into callee sub-graph #ci.
   * */
  TensorIds inSrcs(CalleeIndex ci) const;

  /**
   * The destinations of copies in the callee sub-graph #ci.
   * */
  TensorIds inDsts(CalleeIndex ci) const;

  /**
   * \return The CalleeTensorId #ctId is made up a tensor id (say #tId) and a
   *         callee sub-graph index (say #ci). This method returns true if the
   *         tensor #tId is the destination of a copy into the sub-graph
   *         #ci, from a tensor in this op's sub-graph.
   **/
  bool isDstInCallee(const CalleeTensorId &ctId) const final;

  /**
   * \return true if the callee sub-graph tensor #ctId is copied out from,
   *         when this op completes.
   * */
  bool isSrcInCallee(const CalleeTensorId &ctId) const final;

  /**
   * \return The tensor in the callee sub-graph #ci which is copied from at
   *         output index #o.
   *
   * Note: the current design of the op class assumes that all of the outputs
   * of an op with callees are copied from callee sub-graphs. This is
   * different to inputs, where for example the switch op has a #condition
   * input which is not copied to a sub-graph.
   * */
  TensorId srcInCallee(OutIndex o, CalleeIndex ci) const final;

  /**
   * The CalleeTensorId #ctId is made up of a tensor in this op's sub-graph,
   * say #tId, and a callee sub-graph index, say #ci.
   *
   * \return The tensors in the callee sub-graph #ci to which the tensor #tId
   *         is copied.
   * */
  TensorIds dstsInCallee(const CalleeTensorId &ctId) const final;

  /**
   * \return The input index at which the callee sub-graph tensor #ctId is
   *         copied to at.
   * */
  InIndex inIndex(const CalleeTensorId &ctId) const final;

  /**
   * \return true if the output at index #o is copied out of callee sub-graph
   *         #ci.
   * */
  bool isCopiedOut(OutIndex o, CalleeIndex ci) const final;

  /**
   * \return The output index at which the callee sub-graph tensor #ctId is
   *         copied out of this op.
   * */
  OutIndex outIndex(const CalleeTensorId &) const final;

  void runSim(ISimState &) const final;

  /**
   * Run this op on cpu.
   * */
  virtual void hostRun(const IHostRunner &) const = 0;

  CallEvent event(CalleeIndex i) const {
    return CallEvent(id(), callee(i), i);
  }

  /**
   * The total number of callee sub-graphs of this op. For ops such as Call
   * and Repeat this will be 1, for others like Switch it is more than 1.
   * */
  uint64_t nCallees() const final { return callees_.size(); }

  /**
   * \return true if #sgId is one of this op's callee sub-graphs.
   * */
  bool isCallee(SubGraphId) const;

  SubGraphIds callees() const final { return callees_; }

  SubGraphId callee(CalleeIndex i) const final {
    return callees_.at(i.get());
  }

  /**
   * Zip the input copy sources and destinations together, and return. \sa
   * CopyIn.
   * */
  std::vector<CopyIn> copyIns() const;

  const CopyOuts &outs() const { return outs_; }

  void appendWithCalleesAttributes(std::ostream &) const;

  /**
   * Given that gradients are required for the inputs #fromTargets and
   * gradients for the outputs at indices #inGrads are provided, determine the
   * autodiff objective.
   * */
  virtual poprithms::autodiff::guide::Objective
  localObjective(CalleeIndex,
                 const InIndices &fromTargets,
                 const OutIndices &inGrads) const = 0;

  /**
   * \return true if this op calls a callee multiple times, and the tensor
   *         #tId is a loop carry dependency. Specifically, return true if
   *         #tId is copied to at the end of each iteration, from another
   *         tensor in the callee sub-graph.
   * */
  virtual bool isCarriedTo(const TensorId &tId) const = 0;

  /**
   * The inverse of #isCarriedTo. Returns true if #tId, is a tensor in this
   * op's callee and is copied FROM at the end of each iteration of the
   * callee. This method must return false if this op does not have a callee
   * which is run repeatedly multiple times.
   * */
  virtual bool isCarriedFrom(const TensorId &tId) const = 0;

  /**
   * For ops which repeat a callee sub-graph, return the tensor to which #tId
   * is carried at the end of each iteration. If #tId is not carried (\sa
   * isCarriedTo) then an error is thrown.
   * */
  virtual TensorId carriedFrom(const TensorId &tId) const = 0;

  /**
   * The inverse of #carriedFrom. Specifically, if carriedFrom(#a) is #b then
   * carriedTo(#b) must be #a.
   * */
  virtual TensorId carriedTo(const TensorId &tId) const = 0;

  void computeDerivedVerifyValid() const final;

  /**
   * A method for ops with callees, that do not have loops. It determines, for
   * such an op #wc, if it is possible to traverse from output index #outIndex
   * to input index #inIndex according to the traversal condition #c.
   *
   * For example, the condition #c might return true if an op is
   * differentiable. In this case, the method checks if the gradient at the
   * output index #outIndex can propagate all the way to the input index
   * #inIndex.
   * */
  template <class Condition>
  static bool nonRepeatPropagates(const WithCallees &wc,
                                  OutIndex outIndex,
                                  InIndex inIndex,
                                  const Condition &c);

private:
  /**
   * Only RefFrom_ can have an output which references a tensor in a different
   * sub-graph.
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }

  /**
   * Reset the destination of an in-copy.
   * */
  void resetCalleeTensorId(InIndex, const CalleeTensorId &newId) final;

  /**
   * Reset the source (in a callee sub-graph) of an out-copy.
   * */
  void resetOutSource(OutIndex,
                      CalleeIndex,
                      const TensorId &newSourceInCallee) final;

  void growAliasMapper(MemoryAliasMapper &mam) const final {
    return createVariables(mam);
  }

  /**
   * Ops with callees do execute code at runtime (unlike say the 'VarInit' and
   * 'Reshape_' ops which do not.
   * */
  bool isInitializingOp() const final { return false; }

  bool computeTypeSpecificEqualTo(const Op &) const final;

  /**
   * If all ops in all callees have CodeLocation::Ipu, then this calling op
   * has CodeLocation::Ipu. Otherwise, it has CodeLocation::Host.
   * */
  CodeLocation codeLocation() const final;

  /**
   * Append the relevant callee checkpoint tensors to #ids.
   * */
  void extendAutodiffRequiredTensors(
      poprithms::autodiff::automatic::RequiredIds &ids) const final;

  virtual bool withCalleesTypeSpecificEqualTo(const Op &) const = 0;

  virtual void withCalleesTypeSpecificAssertValid() const = 0;

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final;
  virtual void
  withCalleesDerivedRemoveInputs(const ContiguousInIndexSubset &) = 0;

  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final;
  virtual void
  withCalleesDerivedRemoveOutputs(const ContiguousOutIndexSubset &) = 0;

  /**
   * All outputs are new allocations.
   * */
  HostTensors initializeOut(const HostTensors &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }
};

/**
 * Call operation.
 * */
class Call final : public WithCallees {

public:
  /**
   * \param copyInDestinations The destinations in #callee that the inputs
   *                           are copied to. The sources of the copies are
   *                           the inputs in this op's sub-graph, and defined
   *                           in #state.
   *
   * \param callee The sub-graph which this call op calls.
   *
   * \param copyOutSources The tensors in #callee which are copied out.
   * */
  Call(const State &state,
       const TensorIds &copyInDestinations,
       SubGraphId callee,
       const TensorIds &copyOutSources);

private:
  /**
   * Outputs of call operations are always new allocations. To alias a tensor
   * in the callee sub-graph in the calling op's sub-graph, the RefFrom_ op
   * can be used.
   * */
  bool aliases(InIndex, OutIndex) const final { return false; }
  bool modifies(InIndex) const final { return false; }

  poprithms::autodiff::guide::Objective
  localObjective(CalleeIndex,
                 const InIndices &fromTargets,
                 const OutIndices &inGrads) const final;

  // This operation does not have repeat semantics, and calling these methods
  // is therefore invalid:
  TensorId carriedFrom(const TensorId &) const final { invalid(); }
  TensorId carriedTo(const TensorId &) const final { invalid(); }

  /**
   * As there is only 1 callee sub-graph for a call op, there is only 1
   * CallEvent associated to it.
   * */
  CallEvent event() const {
    return CallEvent(id(), callee(CalleeIndex(0)), CalleeIndex(0));
  }

  /**
   * Create gradient tensors for the inputs of this op.
   * */
  OptionalTensorIds
  growInGrads(Graph &,
              const poprithms::autodiff::core::ToGradGraph &,
              const poprithms::autodiff::automatic::GradInfos &,
              SubGraphId toExtend) const final;

  /**
   * There are no loop-carry dependencies for a call op, as the callee is only
   * called once (unlike a loop-style op).
   * */
  bool isCarriedTo(const TensorId &) const final { return false; }

  bool isCarriedFrom(const TensorId &) const final { return false; }

  void
  withCalleesDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void withCalleesDerivedRemoveInputs(const ContiguousInIndexSubset &) final {
  }

  bool withCalleesTypeSpecificEqualTo(const compute::Op &) const final {
    return true;
  }

  void withCalleesTypeSpecificAssertValid() const final;

  std::string typeString() const final;

  UpOp cloneWithState(const State &) const final;

  void hostRun(const IHostRunner &) const final;

  bool gradientPropagates(OutIndex, InIndex) const final;

  bool isValueDependent(InIndex, OutIndex) const final;
};

/**
 * A repeat operation.
 *
 * This runs a single sub-graph for a fixed number of iterations.
 *
 * Inputs and outputs can be "stacked" or "flat". If they are flat, then they
 *
 * have the same shape in the callee and in the calling scope. If they are
 * stacked, the tensor in the calling scope has an additional prepended
 * dimension which is the repeat count or 'trip count'.
 *
 * Example: if #tInCallee has shape (4,3) and the repeat count is 5, then the
 * corresponding input/output tensor in the calling sub-graph has shape
 * (5,4,3) if it is a stacked input/output, and shape (4,3) if it is a flat
 * input/output.
 *
 * For more information see also the sub-graph API (SubGraph::repeat).
 * */
class Repeat : public WithCallees {

public:
  /**
   * Construct a repeat op.
   * */
  Repeat(const State &s,
         SubGraphId callee,
         uint64_t repeatCount,
         const TensorIds &copyInDestinations,
         const TensorIds &copyOutSources,
         const TensorIds &carriedFrom,
         const TensorIds &carriedTo,
         StackedCopyOrder sto)
      : WithCallees(
            s,
            {callee},
            CalleeTensorId::zip(copyInDestinations, CalleeIndex(0)),
            CopyOuts(std::map<CalleeIndex, TensorIds>{{0, copyOutSources}})),
        repeatCount_(repeatCount), carriedFroms_(carriedFrom),
        carriedTos_(carriedTo), sto_(sto) {}

  ~Repeat() override = default;

  /**
   * With only one callee sub-graph, there is a unique CallEvent assocated to
   * a repeat operation.
   */
  CallEvent event() const;

  bool inputIsStackedCopy(InIndex i) const {
    return inShape(i) != computeGraph().shape(dstInCallee(i).tId());
  }

  bool outputIsStackedCopy(OutIndex o) const {
    return outShape(o) != computeGraph().shape(srcInCallee(o, 0));
  }

  /**
   * This method returns false if the tensor #tId in the callee sub-graph
   * might be different between iterations. It is conservative, in that it
   * might not always return true when the tensor has the same value between
   * iterations.
   */
  bool definitelySameValueEveryIteration(const TensorId &) const;

  uint64_t nCarriedTensors() const { return carriedFroms_.size(); }

  /**
   * The input indices of carried inputs.
   * */
  InIndices carriedInIndices() const;
  bool isCarriedIn(InIndex) const;

  /**
   * The input indices of stacked inputs.
   * */
  InIndices stackedInIndices() const;
  bool isStackedIn(InIndex) const;

  /**
   * \return true if any of the inputs is stacked.
   * */
  bool hasStackedInIndices() const;

  /**
   * The flat output indices.
   * */
  OutIndices flatOutIndices() const;

  /**
   * The stacked output indices.
   * */
  OutIndices stackedOutIndices() const;

  /**
   * \return true if any of the outputs is stacked.
   * */
  bool hasStackedOutIndices() const;

  /**
   * \return true if the output at index #o is flat.
   * */
  bool isFlatOut(OutIndex o) const;

  /**
   * \return true if the output at index #o is stacked.
   * */
  bool isStackedOut(OutIndex o) const { return !isFlatOut(o); }

  /**
   * \return true if the tensor #tId is an output, and moreover it is flat.
   * */
  bool isFlatOut(const TensorId &tId) const;

  /**
   * \return true if the tensor #tId is an output, and moreover it is stacked.
   * */
  bool isStackedOut(const TensorId &) const;

  StackedCopyOrder stackedCopyOrder() const { return sto_; }

  void switchStackedCopyOrder() {
    sto_ = (stackedCopyOrder() == StackedCopyOrder::Down)
               ? StackedCopyOrder::Up
               : StackedCopyOrder::Down;
  }

  /**
   * \return The carried tensor which is copied to #carriedTo.
   * */
  TensorId carriedFrom(const TensorId &carriedTo) const final {
    return carriedFroms_[getIndexInCarriedTo(carriedTo)];
  }
  bool isCarriedFrom(const TensorId &) const final;

  /**
   * \return The tensors, 1 for each tensor in #carriedTos, which are copied
   *         from.
   * */
  TensorIds carriedFroms(const TensorIds &carriedTos) const;

  /**
   * \return The carried tensor which is copied from #carriedFrom.
   * */
  TensorId carriedTo(const TensorId &carriedFrom) const final {
    return carriedTos_[getIndexInCarriedFrom(carriedFrom)];
  }
  bool isCarriedTo(const TensorId &) const final;

  /** The number of times the callee is executed. */
  uint64_t repeatCount() const { return repeatCount_; }

  /**
   * \return The tensors which are traversed through differentiable ops
   *         between in the inputs #inIndices and #outIndices.
   *
   * \sa RepeatDifferentiator::gradientPropagationVisits.
   * */
  TensorIds gradientPropagationVisits(const InIndices &inIndices,
                                      const OutIndices &outIndices) const;

private:
  std::string repeatString() const;

  template <class Accept>
  std::set<TensorId> visitedFwdFrom(const TensorIds &, const Accept &) const;

  template <class Accept>
  std::set<TensorId> visitedBwdFrom(const TensorIds &, const Accept &) const;

  std::set<TensorId> gradientPropagatesFwdFrom(const InIndices &) const;

  std::set<TensorId> gradientPropagatesBwdFrom(const OutIndices &) const;

  virtual poprithms::autodiff::guide::Objective
  localObjective(CalleeIndex,
                 const InIndices &fromTargets,
                 const OutIndices &inGrads) const final;

  // This method effectively unrolls the callee sub-graph to check if a
  // gradient can propagate from #o to #i.
  bool gradientPropagates(OutIndex, InIndex) const final;

  bool isValueDependent(InIndex, OutIndex) const final;

  // Outputs are new allocations, so there is no input-output aliasing.
  bool aliases(InIndex, OutIndex) const final { return false; }

  // No inputs are modified by a repeat operation.
  bool modifies(InIndex) const final { return false; }

  void
  withCalleesDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void withCalleesDerivedRemoveInputs(const ContiguousInIndexSubset &) final;

  bool withCalleesTypeSpecificEqualTo(const Op &) const final;

  void withCalleesTypeSpecificAssertValid() const final;

  // The index in carriedFroms_ of #tId.
  uint64_t getIndexInCarriedFrom(const TensorId &tId) const;

  // The index in carriedTos_ of #tId.
  uint64_t getIndexInCarriedTo(const TensorId &tId) const;

  // If unstacked has shape #s, verify that stacked has shape (rptCount, *s).
  void verifyFirstIsSecondStacked(const TensorId &stacked,
                                  const TensorId &unstacked) const;

  std::string typeString() const final;

  UpOp cloneWithState(const State &) const final;

  void hostRun(const IHostRunner &) const final;

  OptionalTensorIds
  growInGrads(Graph &,
              const poprithms::autodiff::core::ToGradGraph &,
              const poprithms::autodiff::automatic::GradInfos &,
              SubGraphId toExtend) const final;

private:
  // The number of times the callee is run.
  uint64_t repeatCount_;

  // For loop carry tensors, the sources of the carries (in the callee
  // sub-graph).
  TensorIds carriedFroms_;

  // For loop carry tensors, the destinations of the carries (in the callee
  // sub-graph).
  TensorIds carriedTos_;

  // Stacked tensors can be iterated through by lowest-to-highest by or
  // highest-to-lowest index. This enum controls this direction.
  StackedCopyOrder sto_;
};

/**
 * Switch operation.
 *
 * The operation has multiple inputs, the last of which is the conditional
 * tensor that determines which sub-graph is run.
 *
 * All inputs other than the conditional tensor are copied to one callee
 * index.
 *
 * The outputs are all copied from tensors in callee sub-graph which is run,
 * according to the conditional tensor. These copies are optional : it is
 * possible to have no callee tensor specified for an (OutIndex, CalleeIndex)
 * pair.
 * */
class Switch final : public WithCallees {
public:
  Switch(const State &s,
         const SubGraphIds &callees,
         const CalleeTensorIds &inDsts,
         const CopyOuts &copyOuts);

  /**
   * The input index of the condition tensor. It is the final input.
   * */
  InIndex conditionInIndex() const { return nInTensors() - 1; }
  TensorId conditionId() const { return inTensorId(conditionInIndex()); }

  std::string typeString() const final;

  UpOp cloneWithState(const State &) const final;

  void hostRun(const IHostRunner &) const final;

private:
  poprithms::autodiff::guide::Objective
  localObjective(CalleeIndex,
                 const InIndices &fromTargets,
                 const OutIndices &inGrads) const final;

  OptionalTensorIds
  growInGrads(Graph &,
              const poprithms::autodiff::core::ToGradGraph &,
              const poprithms::autodiff::automatic::GradInfos &,
              SubGraphId toExtend) const final;

  bool gradientPropagates(OutIndex, InIndex) const final;

  /**
   * Outputs are new allocations.
   * */
  bool aliases(InIndex, OutIndex) const final { return false; }
  bool modifies(InIndex) const final { return false; }

  /**
   * Carrying tensors is for ops which repeatedly run a callee, which switch
   * does not.
   * */
  TensorId carriedFrom(const TensorId &) const final { invalid(); }
  bool isCarriedTo(const TensorId &) const final { return false; }

  TensorId carriedTo(const TensorId &) const final { invalid(); }
  bool isCarriedFrom(const TensorId &) const final { return false; }

  /**
   * The switch op adds no new attributes on input or output tensors, so these
   * removal methods do nothing.
   * */
  void withCalleesDerivedRemoveInputs(const ContiguousInIndexSubset &) final {
  }
  void
  withCalleesDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  /**
   * The switch op adds no new attributes to its base class, so there are
   * attributes to compare here (and so the comparison returns true, as "empty
   * set" = "empty set").
   * */
  bool withCalleesTypeSpecificEqualTo(const Op &) const final { return true; }

  void withCalleesTypeSpecificAssertValid() const final;

  bool isValueDependent(InIndex, OutIndex) const final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
