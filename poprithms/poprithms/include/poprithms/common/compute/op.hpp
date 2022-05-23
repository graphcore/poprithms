// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OP_HPP
#define POPRITHMS_COMMON_COMPUTE_OP_HPP

#include <poprithms/autodiff/core/togradgraph.hpp>
#include <poprithms/common/compute/device.hpp>
#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/common/compute/initialvalues.hpp>
#include <poprithms/common/compute/simtensormap.hpp>
#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/program/distributed/codelocation.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::ContiguousInIndexSubset;
using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::memory::nest::DisjointRegions;
using poprithms::memory::nest::Region;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::Dimensions;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::ndarray::TensorInfo;
using poprithms::ndarray::TensorInfos;
using poprithms::program::callstack::CalleeIndex;
using poprithms::program::callstack::CalleeTensorId;
using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;
using poprithms::program::distributed::CodeLocation;

class Graph;

using namespace poprithms::common;

using Lower = Shape::Lower;
using Upper = Shape::Upper;

using Shapes = std::vector<Shape>;
using DTypes = std::vector<DType>;

// An op in a graph
class Op : public schedulable::Op {

public:
  /** All Op member variables */
  struct State {

  public:
    State(const schedulable::Op::State &baseState_,
          const DTypes &outDTypes_,
          const DeviceIds &outDeviceIds_,
          const std::vector<CallEvents> &inCopies_,
          const std::vector<CallEvents> &outCopies_,
          const InitialValues &initVals_,
          const std::vector<TensorIds> &derivedRefs_)
        : baseState(baseState_), outDTypes(outDTypes_),
          outDeviceIds(outDeviceIds_), inCopies(inCopies_),
          outCopies(outCopies_), initVals(initVals_),
          derivedRefs(derivedRefs_) {}

    /**
     * Extends the base State with starting attributes for this inheritance
     * layer. In particular, this State has no copies to or from the output
     * tensors of the op, and no derived reference tensors.
     * */
    static State getStartingState(OpId,
                                  SubGraphId,
                                  const TensorIds &ins,
                                  const TensorInfos &outs,
                                  const Graph &);

    const schedulable::Op::State baseState;

    /**
     * The numerical type of each of the outputs of this op.
     * */
    const DTypes outDTypes;

    /**
     * The device which each of the output tensors is on.
     * */
    const DeviceIds outDeviceIds;

    /**
     * All of the call events in the graph which involve a copy from a
     * calling sub-graph into an output tensor of this op. This happens when
     * this op's sub-graph is the callee.
     *
     * Specifically, inCopies[outIndex] is the call events where there is a
     * copy to into this op's outIndex'th output.
     * */
    const std::vector<CallEvents> inCopies;

    /**
     * All of the call events in the graph which involve a copy from this an
     * output of this op to the calling sub-graph. This happens when this
     * op's sub-graph is the callee.
     * */
    const std::vector<CallEvents> outCopies;

    /**
     * The initial values of the output tensors of this (state's) op. These
     * replicated values can only be set for tensors which have
     * DeviceType::Ipu.
     * */
    const InitialValues initVals;

    /**
     * In common::compute::Graphs, tensors may have aliases in multiple
     * sub-graphs. This non-SSA feature makes it possible to describe poplar
     * graphs and programs directly.
     *
     * By default however, tensors are not global. By default, all of the
     * tensors which alias a tensor are in its sub-graph. A user must
     * explicity "opt-in" for cross-graph aliasing via a special type of
     * inputless op (more on this later).
     *
     * The semantics for this feature are as follows.
     *
     * The tensors form a partitioning, where tensors in a equivalence class
     * are all the same underlying tensor, but with different ids and
     * belonging to different sub-graphs. There is one canonical
     * representative in each equivalence class which we call the 'root
     * reference'. In the case where there is no aliasing between sub-graphs,
     * each of the equivalence classes will be of size 1, and every tensor is
     * its own root reference.
     *
     * In an equivalence class of size N there is 1 canonical representative
     * (the root reference) and the other N-1 tensors are called the 'derived
     * references'.
     *
     * This vector stores the derived references of all outputs of this op
     * which are root tensors. If there is no cross-graph aliasing, then there
     * are no derived references.
     * */
    const std::vector<TensorIds> derivedRefs;

    Shape inShape(uint64_t i) const { return baseState.baseState.inShape(i); }

    bool operator==(const State &rhs) const;
    bool operator!=(const State &rhs) const { return !operator==(rhs); }
  };

  Op(const State &s);

  Op &operator=(const Op &) = default;
  virtual ~Op() override    = default;
  Op(const Op &)            = default;
  Op(Op &&)                 = default;
  Op()                      = delete;

  /**
   * Get the State of this compute::Op.
   * */
  State getComputeState() const;

  /**
   * This op does not store its input tensor types, so this call will go via
   * this op's graph -- ops only store their output types.
   * */
  DType inDType(InIndex i) const;

  /**
   * The output type of this op's #o'th output.
   * */
  DType outDType(OutIndex o) const { return outDTypes_.at(o.get()); }

  /**
   * The numerical type of the input/output (depending on #p) at index #i.
   * */
  DType dtype(Port p, uint64_t i) const;

  /**
   * The graph to which this op belongs.
   * */
  const Graph &computeGraph() const;

public:
  void insertInCopy(OutIndex, const CallEvent &);
  void insertOutCopy(OutIndex, const CallEvent &);

  void removeInCopy(OutIndex, const CallEvent &);
  void removeOutCopy(OutIndex, const CallEvent &);

  /**
   * All call events which begin with a copy into the #o'th output tensor of
   * this op.
   * */
  const CallEvents &inCopies(OutIndex o) const {
    return inCopies_.at(o.get());
  }

  /**
   * All call events which begin with a copy into one of this op's output
   * tensors, from a tensor in the calling graph (this op is in a callee
   * graph).
   * */
  const std::vector<CallEvents> &inCopies() const { return inCopies_; }

  /**
   * The sub-graphs that this op calls into (if any). For most 'normal' ops
   * this will be the empty vector, for an if-op this will be the 2 sub-graphs
   * (if and else branches), etc.
   * */
  virtual SubGraphIds callees() const = 0;

  /**
   * Callee #ci.
   * */
  virtual SubGraphId callee(CalleeIndex ci) const = 0;

  /**
   * The input index at which the callee tensor #ctId is copied into.
   * Specifically, this method is used when this op has callee sub-graphs, and
   * #ctId is a tensor in one of the callee sub-graphs to which an input of
   * this op (in the calling graph) is copied.
   **/
  virtual InIndex inIndex(const CalleeTensorId &ctId) const = 0;

  /**
   * The output index at which the callee tensor #ctId is copied out of one of
   * this op's callee sub-graphs.
   * */
  virtual OutIndex outIndex(const CalleeTensorId &ctId) const = 0;

  /**
   * The number of callee sub-graphs that this op has.
   * */
  virtual uint64_t nCallees() const = 0;

  bool hasCallees() const { return nCallees() != 0; }

  /**
   * The callees, tied to their callee indices. For example if this op has
   * callees (2,5) then this method returns ((2,0),(5,1)).
   * */
  std::vector<std::pair<SubGraphId, CalleeIndex>> indexedCallees() const;

  /**
   * All call events which end with a copy from the #o'th output tensor of
   * this op into the calling graph.
   * */
  const CallEvents &outCopies(OutIndex o) const {
    return outCopies_.at(o.get());
  }

  const std::vector<CallEvents> &outCopies() const { return outCopies_; }

  /**
   * Remove the inputs of this op at the indices defined by #insToRemove.
   * */
  void computeOpRemoveInputs(const ContiguousInIndexSubset &insToRemove);

  /**
   * Remove the outputs of this op at the indices defined by #insToRemove.
   * */
  void computeOpRemoveOutputs(const ContiguousOutIndexSubset &);

  /**
   * The device of the input at index #i.
   * */
  DeviceId inDeviceId(InIndex i) const;

  /**
   * The device of the output at index #o.
   * */
  DeviceId outDeviceId(OutIndex o) const { return outDeviceIds_.at(o.get()); }

  /**
   * The input or output device at index i.
   * */
  DeviceId deviceId(Port, uint64_t i) const;

  /**
   * The devices of all of the inputs.
   * */
  DeviceIds inDeviceIds() const;

  /**
   * The devices of all of the outputs.
   * */
  DeviceIds outDeviceIds() const { return outDeviceIds_; }

  /**
   * The tensor information (shape, type, device) of input #i.
   * */
  TensorInfo inTensorInfo(InIndex i) const;

  /**
   * The tensor information (shape, type, device) of output #o.
   * */
  TensorInfo outTensorInfo(OutIndex o) const;

  /**
   * The tensor information (shape, type, device) of all inputs.
   * */
  TensorInfos inTensorInfos() const;

  /**
   * The tensor information (shape, type, device) of all outputs.
   * */
  TensorInfos outTensorInfos() const;

  /**
   * Verify that all the attributes at this level of op inheritance are valid,
   * for this op.
   * */
  void verifyValidAtComputeLevel() const;

  /**
   * Get the initial values (if any) for each replica if the output tensor #o.
   * */
  std::map<uint64_t, poprithms::compute::host::Tensor>
  initialValues(OutIndex o) const;

  /**
   * Set the initial value of replica #r of output tensor #o to #val. This can
   * only be done for tenors with DeviceType::Ipu.
   * */
  void setInitialValue(uint64_t r,
                       OutIndex o,
                       const poprithms::compute::host::Tensor &val);

  /**
   * The device type of the input tensor at index #i.
   * */
  DeviceType inDeviceType(InIndex i) const;

  /**
   * The device type of the output tensor #o.
   * */
  DeviceType outDeviceType(OutIndex o) const;

  bool isIpu(OutIndex o) const { return outDevice(o).isIpu(); }
  bool isHost(OutIndex o) const { return outDevice(o).isHost(); }
  bool isRemove(OutIndex o) const { return outDevice(o).isRemote(); }

  /**
   * Return true if there is at least one input/output tensor which is on
   * host, and at least one which is not.
   * */
  bool isPartiallyHost() const;

  /**
   * The device types of all of the input tensors.
   * */
  DeviceTypes inDeviceTypes() const;

  /**
   * The device types of all of the output tensors.
   * */
  DeviceTypes outDeviceTypes() const;

  /**
   * A concatenation of all of the input and output device types. If this op
   * has M input tensors and N output tensors, the returned vector contains M
   * + N elements.
   * */
  DeviceTypes inAndOutDeviceTypes() const;

  /**
   * The device of the output tensor at index #o.
   * */
  const Device &outDevice(OutIndex o) const;

  /**
   * The device of the input tensor at index #i.
   * */
  const Device &inDevice(InIndex i) const;

  /**
   * The type of the device of the input/output tensor at index #i.
   * */
  DeviceType deviceType(Port, uint64_t i) const;

  /**
   * The device of input/output at index #i.
   * */
  const Device &device(Port, uint64_t) const;

  /**
   * The device type of this op, inferred from the device types of all inputs
   * and outputs. If not all inputs and outputs have the same device type, an
   * error is thrown.
   * */
  DeviceType deviceTypeByUnanimity() const;

  /**
   * The root reference tensor for output tensor #i. In other words, the
   * canonical representative of the equivalence class of identical tensors in
   * different sub-graphs.
   *
   * If output #o does not have references in other graphs, this will be the
   * output #o itself.
   * */
  virtual TensorId rootRef(OutIndex o) const = 0;

  bool isRootRef(OutIndex o) const { return rootRef(o) == outTensorId(o); }

  /**
   * If the output #o is a root reference tensor (the canonical representative
   * of its equivalence class), return then N-1 other tensors in the
   * equivalence class (where N is the size of the equivalence class). If
   * output #o is not a root reference, return {}.
   * */

  TensorIds derivedRefs(OutIndex o) const { return derivedRefs_.at(o.get()); }

  uint64_t nDerivedRefs(OutIndex o) const {
    return derivedRefs_.at(o.get()).size();
  }

  bool hasDerivedRefs(OutIndex o) const { return nDerivedRefs(o) != 0; }

  /**
   * All tensors in the equivalence class of output #o formed of identical
   * tensors in different sub-graphs, excluding output #o.
   * */
  TensorIds refsExcludingSelf(OutIndex) const;

  void insertOutDerivedRef(OutIndex, TensorId);

  /**
   * Return true if the input at index #i is fixed point (integral).
   * */
  bool inIsFixedPoint(InIndex i) const;

  /**
   * Return true if the output at index #o is fixed point (integral).
   * */
  bool outIsFixedPoint(OutIndex o) const;

  /**
   * If this op performs zero compute cycles, it is an 'initializing op'.
   * Examples are view-changing ops (reshape, slice, etc.) without any data
   * copies, and ops which initialize constants and variables.
   * */
  virtual bool isInitializingOp() const = 0;

  /**
   * Initializing ops can appear anywhere in a schedule.
   * */
  bool isConstraintPhobic() const final { return isInitializingOp(); }

public:
  /**
   * Update the tensors in #simTensors corresponding to the output tensors of
   * this op, by running this op on cpu.
   * */
  virtual void runSim(SimTensorMap &simTensors) const = 0;

  /**
   * Initialize the tensors in #simTensors corresponding to the output tensors
   * of this op.
   * */
  virtual void initializeSimOut(SimTensorMap &) const = 0;

  /**
   * Initialize the output tensors of this op, based on the input tensors
   * #ins.
   * */
  virtual HostTensors initializeOut(const HostTensors &ins) const = 0;

  /**
   * A utility method for creating output tensors for this op with the value
   * 0.
   * */
  HostTensors zeroOuts() const;

  /**
   * A utility method for creating output tensors for this op with non-0
   * values.
   * */
  HostTensors badValOuts() const;

  virtual CodeLocation codeLocation() const = 0;

protected:
  /**
   * A utility method for initializing ipu tensors in a SimTensorMap. This
   * method initializes the tensors in #simTensors corresponding to the
   * outputs of this op.
   *
   * This method is protected, as it is only ever called into by op
   * implementations of initializeSimOut.
   * */
  void initializeReplicatedSimOut(SimTensorMap &simTensors) const;

  CodeLocation locationByUnanimity() const;

private:
  const Graph &graph() const { return computeGraph(); }

  bool schedulableTypeSpecificEqualTo(
      const poprithms::common::schedulable::Op &other) const final;

  /**
   * A pure virtual function that derived classes must implement.
   * This function has a precondition that it will only
   * be called when the 'other' is the same type as the instance
   * invoking the function.
   * */
  virtual bool computeTypeSpecificEqualTo(const Op &other) const = 0;

  // These are methods defined in schedulable::Op. We don't want them to be
  // publicly accessible at this level, all constraints should be inserted
  // with Graph::constraint. Hence making them private.
  void insertIn(OpId);
  void insertOut(OpId);

  using poprithms::common::multiout::Op::multioutGraph;

private:
  // See the comments in the Op::State class about these attributes.
  DTypes outDTypes_;
  DeviceIds outDeviceIds_;
  std::vector<CallEvents> inCopies_;
  std::vector<CallEvents> outCopies_;
  // indexed as [out index][replica]:
  InitialValues initVals_;
  std::vector<TensorIds> derivedRefs_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
