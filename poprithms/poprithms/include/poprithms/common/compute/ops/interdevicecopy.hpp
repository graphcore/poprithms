// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_INTERDEVICECOPY_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_INTERDEVICECOPY_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withautodiff.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * A base class for copies in of tensors between the host device and an ipu
 * device. The op has 2 inputs, a source and destination of the copy. It has 1
 * output, which is an alias of the destination of the copy.
 *
 * If the ipu-side tensor has rank #R, then the host-side tensor has rank
 * #R+2, with the 2 additional dimensions accounting for circular buffer size
 * and replication.
 * */
class CopyBetweenHostAndIpu_ : public WithoutCalleesTensorCentric {

public:
  /**
   * \param options_ Any additional options describing how the copy is
   *                 performed.
   **/
  CopyBetweenHostAndIpu_(const Op::State &,
                         const CopyBetweenHostAndIpuOptions &options_);

  /**
   * The source of the copy.
   * */
  static constexpr int SourceIndex{0};
  static InIndex Source() { return SourceIndex; }
  TensorId sourceId() const { return inTensorId(Source()); }
  Shape sourceShape() const { return inShape(Source()); }

  /**
   * The value of the output only depends on the source of the copy. It is
   * independent of the value of input which is the destination of the copy.
   * */
  bool isValueDependent(InIndex i, OutIndex) const final {
    return i == Source();
  }

  /**
   * The destination of the copy.
   * */
  static constexpr int DestinationIndex{1};
  static InIndex Destination() { return DestinationIndex; }
  TensorId destinationId() const { return inTensorId(Destination()); }
  Shape destinationShape() const { return inShape(Destination()); }

  /**
   * Return the input index at which the host tensor is received.
   * */
  virtual InIndex hostInputIndex() const = 0;

  /**
   * \return The input index of the ipu tensor. This is the index which is not
   *         the host tensor's input index.
   * */
  InIndex ipuInputIndex() const { return hostInputIndex() == 1 ? 0 : 1; }

  /**
   * The output tensor id. The output is an alias of the destination.
   * */
  TensorId outId() const { return outTensorId(0); }

  const CopyBetweenHostAndIpuOptions &copyOptions() const {
    return copyOptions_;
  }

  /**
   * This size of the host-side cicrular buffer. Recall that the host tensor
   * has shape (CircularBufferSize, ReplicationFactor, *ipuShape).
   * */
  CircularBufferCount circularBufferCount() const {
    return inShape(hostInputIndex()).dim(0);
  }

  std::string handle() const { return str(); }

  TensorId hostInputId() const { return inTensorId(hostInputIndex()); }

private:
  /**
   * The input at the DestinationIndex is an alias of the output.
   * */
  bool aliases(InIndex i, OutIndex) const final { return i == Destination(); }
  bool modifies(InIndex i) const final { return aliases(i, 0); }

  /**
   * The gradient is a copy of the gradient in the reverse direction of this
   * op. That is, if this op is ipu->host the gradient op is host->ipu, etc.
   * The gradient propagates through inIndex=SourceIndex.
   * */
  bool gradientPropagates(OutIndex, InIndex inIndex) const final;
  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  HostTensors initializeOut(const HostTensors &) const final;

  /**
   * Initialize #stm by adding the output tensor of this op and a counter for
   * the position of the circular buffer (this is required to emulate poplar's
   * circular buffer).
   * */
  void initializeSimOut(SimTensorMap &stm) const final;

  /**
   * The output is an alias of the destination input.
   * */
  void growAliasMapper(MemoryAliasMapper &mam) const final {
    createAlias(mam, destinationId());
  }

  /**
   * Copy data for all replicas, and increment the circular buffer counter.
   * */
  void runSim(ISimState &ss) const final;

  /**
   * Invalid as runSim is implemented directly.
   * */
  void compute(const HostTensors &, const HostTensors &) const final;

  /**
   * Copy to/from host for replica #replica when the circular buffer is at
   * #cci. Called into by #runSim.
   * */
  virtual void runCopyHostSim(const HostTensors &src,
                              const HostTensors &dst,
                              uint64_t replica,
                              uint64_t cci) const = 0;

  /**
   * The copy op does 'computation' and so it is not an 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}

  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  /**
   * Multiple assertions that the shape of the ipu and host tensors are
   * compatible.
   * */
  void computeDerivedVerifyValid() const final;

  uint64_t bufferingDepth() const { return copyOptions_.bufferingDepth(); }

  bool computeTypeSpecificEqualTo(const compute::Op &rhs) const final;

  // The ops which copy between ipu and host are edge cases for code location,
  // as the 2 inputs have different device types. However, as the copy does
  // require poplar code, they are defined to be on Ipu instead of Host.
  CodeLocation codeLocation() const final { return CodeLocation::Ipu; }

  /**
   * The output tensor is an alias of the destination input.
   * */
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  CopyBetweenHostAndIpuOptions copyOptions_;
};

/**
 * Copy from host to ipu.
 * */
class CopyFromHostToIpu_ final : public CopyBetweenHostAndIpu_ {
public:
  CopyFromHostToIpu_(const State &, const CopyBetweenHostAndIpuOptions &);

  bool isBroadcast() const { return sourceShape().dim(1) == 1; }

private:
  UpOp cloneWithState(const State &) const final;

  std::string typeString() const final { return "CopyFromHostToIpu_"; }

  void runCopyHostSim(const HostTensors &src,
                      const HostTensors &dst,
                      uint64_t replica,
                      uint64_t currentCircularBufferIndex) const final;

  OptionalTensors bprop(const GradOpIns &) const final;

  InIndex hostInputIndex() const final { return Source(); }
};

/**
 * Copy from ipu to host.
 * */
class CopyFromIpuToHost_ final : public CopyBetweenHostAndIpu_ {
public:
  CopyFromIpuToHost_(const State &, const CopyBetweenHostAndIpuOptions &);

private:
  UpOp cloneWithState(const State &) const final;

  std::string typeString() const final { return "CopyFromIpuToHost_"; }

  void runCopyHostSim(const HostTensors &src,
                      const HostTensors &dst,
                      uint64_t replica,
                      uint64_t currentCircularBufferIndex) const final;

  OptionalTensors bprop(const GradOpIns &) const final;

  InIndex hostInputIndex() const final { return Destination(); }
};

/**
 * Copy between remote and ipu.
 *
 * This op has 3 inputs:
 *   (1) An ipu tensor of type T
 *   (2) A remote tensor of type T
 *   (3) An index tensor (integral, on ipu) which defines which part of the
 *       remote tensor to copy to/from.
 *
 * Ops which inherit from this op define which direction the copy is. Either
 * (1) -> (2) or (2) -> (1).
 *
 * The shapes of the inputs are:
 *
 *  (1) (n0, S)
 *  (2) (n1, S)
 *  (3) (n0)
 *
 *  The op copies n0 slices of (2) to/from the n0 slices of (1).
 * */
class CopyBetweenRemoteAndIpu_
    : public NoAutodiff<WithoutCalleesTensorCentric> {

public:
  CopyBetweenRemoteAndIpu_(const State &s) : NoAutodiff(s) {}

  static InIndex RemoteSliceable() { return InIndex(0); }
  static InIndex IpuSlice() { return InIndex(1); }
  static InIndex Indices() { return InIndex(2); }

  /**
   * Given the shape of the indices and remote tensors, infer the shape of the
   * ipu tensor: (indices.dim(0), remote.dim(1)).
   * */
  static Shape shapeOfIpuSlice(const Shape &indices, const Shape &remote);

  /**
   * Given the shape of the ipu tensor and number of repeats, infer the shape
   * of the remote tensor: (nRepeats, remote.dim(1)).
   * */
  static Shape shapeOfRemoteSliceable(const Shape &ipuSlice,
                                      uint64_t nRepeats);

private:
  /**
   * Methods specific to the RefFrom op:
   * */
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  /**
   * The input which the output is aliased to is modified by this op.
   * */
  bool modifies(InIndex i) const final { return aliases(i, 0); }

  /**
   * Verify that #indicesShape is rank-2.
   * */
  static void verifyIndicesShape(const Shape &indicesShape);

  HostTensors initializeOut(const HostTensors &) const final;

  virtual InIndex aliasInIndex() const = 0;

  /**
   * Explanation for why this op does not (currently) support autodiff.
   * */
  std::string whyNoAutodiff() const final;

  /**
   * The output is an alias of one of the inputs.
   * */
  bool aliases(InIndex i, OutIndex) const final {
    return i == aliasInIndex();
  }

  void computeDerivedVerifyValid() const final;

  void runSim(ISimState &htm) const final {
    runReplicatedSim(htm.simTensorMap());
  }

  CodeLocation codeLocation() const final { return CodeLocation::Ipu; }

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void growAliasMapper(MemoryAliasMapper &b) const final {
    createAlias(b, inTensorId(aliasInIndex()));
  }

  bool isInitializingOp() const final { return false; }
};

/**
 * The ipu tensor (1) is updated inplace with the values copied from (2).
 *
 * The returned tensor is an alias of the ipu tensor which is written to.
 * */
class CopyFromRemoteToIpu_ final
    : public Attributeless<CopyBetweenRemoteAndIpu_, CopyFromRemoteToIpu_> {
public:
  CopyFromRemoteToIpu_(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName{"CopyFromRemoteToIpu_"};

private:
  InIndex aliasInIndex() const final { return IpuSlice(); }

  bool isValueDependent(InIndex i, OutIndex) const final {
    return i != IpuSlice();
  }

  // Expectation of which tensors are remote. This is used in when verifying
  // the op after construction.
  bool supportsRemote(const InIndices &i, const OutIndices &o) const final {
    return i == InIndices{RemoteSliceable()} && o.empty();
  }

  void compute(const HostTensors &, const HostTensors &) const final;
};

/**
 * The remote tensor (2) is updated inplace with the values copied from (1).
 *
 * The returned tensor is an alias of the remote tensor which is written to.
 * */
class CopyFromIpuToRemote_ final
    : public Attributeless<CopyBetweenRemoteAndIpu_, CopyFromIpuToRemote_> {
public:
  CopyFromIpuToRemote_(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName{"CopyFromIpuToRemote_"};

private:
  InIndex aliasInIndex() const final { return RemoteSliceable(); }

  bool isValueDependent(InIndex, OutIndex) const final { return true; }

  bool supportsRemote(const InIndices &i, const OutIndices &o) const final {
    return i == InIndices{RemoteSliceable()} && o == OutIndices{0};
  }

  void compute(const HostTensors &, const HostTensors &) const final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
