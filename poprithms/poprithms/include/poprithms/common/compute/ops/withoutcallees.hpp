// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_WITHOUTCALLEES_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_WITHOUTCALLEES_HPP

#include <poprithms/common/compute/op.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * An op which has no callee sub-graphs.
 * */
class WithoutCallees : public Op {
public:
  WithoutCallees(const Op::State &s) : Op(s) {}

  /**
   * This op has no callee sub-graphs.
   * */
  uint64_t nCallees() const final { return 0; }

  /**
   * This op has no callee sub-graphs.
   * */
  SubGraphIds callees() const final { return {}; }

  /**
   * As there are no callee sub-graphs, none of the inputs is copied to a
   * callee-subgraph.
   * */
  bool isCopyToCalleeInIndex(InIndex) const final { return false; }

  /**
   * These are all invalid calls for ops which have no callee sub-graphs, and
   * will have errors thrown if called.
   *
   * Design note: Having a bunch of virtual methods like this which are
   * overriden to throw errors is not elegant design. An alternative would
   * have been to only introduce these methods in a base class for ops with
   * callees. This would have had its own drawbacks however: unavoidable
   * dynamic_casts in places, and a less complete op class which already has
   * the concept of copies into and out of callees.
   * */
  InIndex inIndex(const CalleeTensorId &) const final;
  OutIndex outIndex(const CalleeTensorId &) const final;
  SubGraphId callee(CalleeIndex) const final;
  CalleeTensorId dstInCallee(InIndex) const final;
  TensorId srcInCallee(OutIndex, CalleeIndex) const final;
  bool isDstInCallee(const CalleeTensorId &) const final;
  bool isSrcInCallee(const CalleeTensorId &) const final;
  TensorIds dstsInCallee(const CalleeTensorId &) const final;
  bool isCopiedOut(OutIndex, CalleeIndex) const final;
  void resetCalleeTensorId(InIndex, const CalleeTensorId &) final;
  void resetOutSource(OutIndex, CalleeIndex, const TensorId &) final;

  /**
   * For most ops without callees, running code for the simulator code follows
   * this chain of calls:
   *
   *    1) runSim.
   *       This is the base method in the op class which is the entry point
   *       for all ops, with and without callees.
   *
   *       --- calls into -->
   *
   *    2) runReplicatedSim.
   *       This method inserts a loop over the replication factor for tensors
   *       which are on ipu.
   *
   *       --- calls into -->
   *
   *    3) computeWithChecks.
   *       This method performs multiple consistency checks on the host
   *       tensors (size, shape, etc.)
   *
   *       --- calls into -->
   *
   *    4) compute.
   *       This method implements the op specific logic for doing arithmetic
   *       on host tensors.
   *
   * Some ops without callees do not follow this call pattern. Examples are
   * ops which copy tensors between the host and ipu, for which
   * 'runReplicatedSim' is not appropriate, and ops which perform reductions
   * across replicas.
   *
   * This method calls into the virtual method 'compute' after performing some
   * checks in #ins and #outs.
   * */

  void computeWithChecks(const HostTensors &ins, HostTensors &outs) const;

  /**
   * Ops without callees only ever require inputs, outputs, and/or gradients
   * of outputs. They therefore only need to implement the simpler methods,
   * #autodiffRequiredIns and #autodiffRequiredOuts.
   * */
  void extendAutodiffRequiredTensors(
      autodiff::automatic::RequiredIds &) const final;

  /**
   * Ops without callees have a simpler time of creating ops in #graph during
   * backpropagation. They therefore only need to implement the simple
   * #backpropagate method.
   * */
  OptionalTensorIds growInGrads(Graph &graph,
                                const ToGradGraph &t,
                                const autodiff::automatic::GradInfos &,
                                SubGraphId) const final;

protected:
  /**
   * This method is protected, as it should only be called by ops in their
   * implementations of runSim.
   * */
  void runReplicatedSim(SimTensorMap &) const;

private:
  /**
   * Perform a computation which updates output values #outs based on the
   * input values #ins. The implementation of this method does not need to
   * perform checks on #ins or #outs because computeWithChecks, the public API
   * method, does.
   * */
  virtual void compute(const HostTensors &ins,
                       const HostTensors &outs) const = 0;

  /**
   * The inputs of this op which are required to perform backpropagation.
   * */
  virtual std::vector<InIndex> autodiffRequiredIns() const = 0;

  /**
   * The outputs of this op which are required to perform backpropagation.
   * */
  virtual std::vector<OutIndex> autodiffRequiredOuts() const = 0;

  /**
   * Propagate the gradient(s) of the outputs of this op. The gradients of the
   * outputs, and the inputs and outputs required, are available in
   * #gradOpInIds.
   *
   * \return The gradients of the input tensors of this op. Some inputs are
   *         not on a non-zero differentiable path to any output, these must
   *         have unset OptionalTensorIds returned.
   * */
  virtual OptionalTensorIds
  backpropagate(Graph &graph, const GradOpInIds &gradOpInIds) const = 0;

  [[noreturn]] void invalidAsNoCallees() const;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
