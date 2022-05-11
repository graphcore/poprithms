// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OP_HPP
#define POPRITHMS_COMMON_COMPUTE_OP_HPP

#include <poprithms/autodiff/core/togradgraph.hpp>
#include <poprithms/common/compute/device.hpp>
#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/common/compute/initialvalues.hpp>
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
#include <poprithms/program/callstack/callstack.hpp>

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
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;

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
          const InitialValues &initVals_)
        : baseState(baseState_), outDTypes(outDTypes_),
          outDeviceIds(outDeviceIds_), inCopies(inCopies_),
          outCopies(outCopies_), initVals(initVals_) {}

    /**
     * Extends the base State with starting attributes for this inheritance
     * layer. In particular, this State has no copies to or from the output
     * tensors of the op.
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
  const Graph &graph() const;

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

  // if in and out don't all agree, throws error
  DeviceType deviceType() const;

protected:
  /**
   * Some utility methods used for checking for correctness of ops where
   * there is an expectation on the equivalence of input/output types.
   * */
  void verifyInsSameDType() const;
  void verifyOutsSameDType() const;
  void verifyAllSameDType() const;

private:
  // See the comments in the Op::State class about these attributes.
  DTypes outDTypes_;
  DeviceIds outDeviceIds_;
  std::vector<CallEvents> inCopies_;
  std::vector<CallEvents> outCopies_;
  // indexed as [out index][replica]:
  InitialValues initVals_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
