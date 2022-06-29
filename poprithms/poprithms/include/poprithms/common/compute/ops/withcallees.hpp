// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_WITHCALLEES_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_WITHCALLEES_HPP

#include <poprithms/common/compute/ihostrunner.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copyout.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::program::callstack::CopyIn;
using poprithms::program::callstack::CopyOuts;

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
   * \return The number of inputs which are not copied to callee sub-graphs.
   * */
  virtual uint64_t nNonCopyIns() const = 0;

  /**
   * \return The number of inputs which are copied to callee sub-graphs.
   * */
  uint64_t nCopyIns() const { return nInTensors() - nNonCopyIns(); }

  /**
   * All the input copies are at input indices lower than the non-copy inputs.
   * */
  bool isCopyToCalleeInIndex(InIndex i) const final {
    return i.get() < nCopyIns();
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
   * \param i An input index, must be less than #nCopyIns.
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

  void runSim(SimTensorMap &) const final;

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

  void computeDerivedVerifyValid() const final;

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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
