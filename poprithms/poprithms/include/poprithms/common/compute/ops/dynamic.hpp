// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_DYNAMIC_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_DYNAMIC_HPP

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
 * An op with 3 inputs.
 *
 * 1) Sliceable tensor.
 *    A tensor from which a region (sub-tensor) is selected dynamically. The
 *    region has a fixed shape, but its position within the sliceable tensor
 *    is dynamic, being determined at runtime by the values in the offset
 *    tensor.
 *
 * 2) Offset tensor.
 *    Indices which define the position of the dynamic region in the sliceable
 *    tensor.
 *
 * 2) Slice.
 *    A tensor whose shape matches the dynamic region in the sliceable
 *    tensor, in all but some batch/group dimensions. i.e. in the "spatial"
 *    dimensions.
 *
 * Ops which inherit from this op either,
 * - update the slice tensor inplace, or
 * - update the sliceable tensor inplace.
 */
class DynamicMulti : public WithoutCalleesTensorCentric {
public:
  DynamicMulti(const State &s) : WithoutCalleesTensorCentric(s) {}

  /**
   * The input indices of the 3 tensors described above.
   * */
  const static InIndex Sliceable() { return 0; }
  const static InIndex Slice() { return 1; }
  const static InIndex Offset() { return 2; }

  /**
   * The tensor ids of the 3 inputs.
   * */
  TensorId sliceableInId() const { return inTensorId(Sliceable()); }
  TensorId sliceInId() const { return inTensorId(Slice()); }
  TensorId offsetId() const { return inTensorId(Offset()); }

  /**
   * The shapes of the 3 inputs.
   * */
  Shape sliceableShape() const { return inShape(Sliceable()); }
  Shape sliceShape() const { return inShape(Slice()); }
  Shape offsetShape() const { return inShape(Offset()); }

  /**
   * The input index at which the output tensor is an alias. For some ops, the
   * output is an alias of sliceable tensors, and for others it is an alias of
   * the slice tensor.
   * */
  virtual InIndex aliasIndex() const = 0;

private:
  CodeLocation codeLocation() const final { return locationByUnanimity(); }
  bool aliases(InIndex i, OutIndex) const final { return i == aliasIndex(); }
  bool modifies(InIndex i) const final { return aliases(i, OutIndex(0)); }

  void runSim(ISimState &ss) const final {
    runReplicatedSim(ss.simTensorMap());
  }

  /**
   * This op does computation, it is therefore not an 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  /**
   * Methods which initialize the output (of some op in some graph) to be an
   * alias of the input at the aliasing input index.
   * */
  HostTensors initializeOut(const HostTensors &ins) const final {
    return {ins.at(aliasIndex().get())};
  }
  void growAliasMapper(MemoryAliasMapper &mam) const final {
    createAlias(mam, inTensorId(aliasIndex()));
  }
  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  /**
   * This op is not a RootRef_.
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
};

/**
 * Dynamic update with a maximum element.
 *
 * The inputs are
 *
 * Sliceable of shape (M, S)
 * Slice     of shape (N, S)
 * Offsets   of shape (S).
 *
 * Sliceable is updated by inplace-maximization with S-vectors from Slice, at
 * indices defined of Offsets. See Tensor::dynamicMultiUpdateMax_ for more
 * information.
 * */
class DynamicMultiUpdateMax_ final
    : public Attributeless<DynamicMulti, DynamicMultiUpdateMax_> {
public:
  DynamicMultiUpdateMax_(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "DynamicMultiUpdateMax_";

private:
  void computeDerivedVerifyValid() const final;

  InIndex aliasIndex() const final { return Sliceable(); }

  void compute(const HostTensors &, const HostTensors &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  OptionalTensors bprop(const GradOpIns &) const final;
  std::vector<InIndex> autodiffRequiredIns() const final {
    return {Offset(), Slice()};
  }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {0}; }

  bool gradientPropagates(OutIndex, InIndex i) const final {
    return i == Slice();
  }
};

/**
 * A dynamic op where the sliceable tensors can be of any rank. The
 * relationship between the shapes of the 3 tensors is described in
 * Tensor::dynamicMultiSlice.
 * */
class DynamicMultiWithDimensions_ : public DynamicMulti {

public:
  DynamicMultiWithDimensions_(const State &s, const Dimensions &dims)
      : DynamicMulti(s), dims_(dims) {}

  /**
   * The dimensions which are sliced
   * */
  Dimensions dimensions() const { return dims_; }
  std::vector<uint64_t> dimensions_u64() const { return dims_.get(); }

  /**
   * The sizes of the output, in the sliced dimensions only.
   * */
  Shape sizes() const;
  std::vector<uint64_t> sizes_u64() const;

  static Shape getSlicedShape(const Shape &offsetShape,
                              const Shape &sliceableShape,
                              const Dimensions &dims,
                              const Shape &sizes);

private:
  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }
  bool computeTypeSpecificEqualTo(const compute::Op &rhs) const final;
  void computeDerivedVerifyValid() const final;

private:
  Dimensions dims_;
};

class DynamicMultiUpdate_ final : public DynamicMultiWithDimensions_ {
public:
  DynamicMultiUpdate_(const State &s, const Dimensions &dims);
  InIndex aliasIndex() const final { return Sliceable(); }

private:
  std::string typeString() const final;
  UpOp cloneWithState(const State &) const final;
  void compute(const HostTensors &, const HostTensors &) const final;
  OptionalTensors bprop(const GradOpIns &) const final;

  // The gradient propagates to the non-aliasing input. i.e. the gradient goes
  // to the source of the copy.
  bool gradientPropagates(OutIndex, InIndex i) const final {
    return i == Slice();
  }
  std::vector<InIndex> autodiffRequiredIns() const final {
    return {Offset()};
  }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
};

class DynamicMultiSlice_ final : public DynamicMultiWithDimensions_ {
public:
  DynamicMultiSlice_(const State &s, const Dimensions &dims);
  InIndex aliasIndex() const final { return Slice(); }

private:
  std::string typeString() const final;
  UpOp cloneWithState(const State &) const final;
  void compute(const HostTensors &, const HostTensors &) const final;
  OptionalTensors bprop(const GradOpIns &) const final;
  std::vector<InIndex> autodiffRequiredIns() const final {
    return {Offset()};
  }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  bool gradientPropagates(OutIndex, InIndex i) const final {
    return i == Sliceable();
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
