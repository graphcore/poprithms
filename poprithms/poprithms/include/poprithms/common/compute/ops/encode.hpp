// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_ENCODE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_ENCODE_HPP

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
 * One-hot encoder. It has 2 inputs:
 *  - an 'indices' tensor which defines where to encode with value 'on'.
 *  - a tensor to encode, with value 'on' at positions defined by 'indices',
 *    and value 'off' elsewhere.
 *
 * This abstract base class does not specify the scalar values which will
 * populate the 'on' and the 'off' indices.
 *
 * The tensor to encode must be rank-2 of shape (N, C).
 * The indices tensor, specifiying where the 'on' values go, must shape (N,)
 *
 * The tensor to encode is encoded inplace, and the output is an alias of it.
 *
 * Example:
 * N=2, C=3, indices = [2,1]. off = 0 and on = 1. Then theencoding is:
 *
 *   [[0   0   1]
 *    [0   1   0]].
 *
 * The values in indices must be integers in the range [0, C). This is not
 * something which can be checked at compile time, as these are runtime
 * values.
 * */
class EncodeOneHot_ : public ZeroAutodiff<WithoutCalleesTensorCentric> {
public:
  EncodeOneHot_(const Op::State &s)
      : ZeroAutodiff<WithoutCalleesTensorCentric>(s) {}

  /**
   * The output value is independent of the input value at index ToEncode, as
   * the value gets overwritten.
   * */
  bool isValueDependent(InIndex i, OutIndex) const final {
    return i == Indices();
  }

  /**
   * The input index of the tensor to encode. The tensor is encoded inplace,
   * and is populated with 2 values, an 'on' and an 'off' value.
   * */
  static InIndex ToEncode() { return 0; }

  /**
   * The input index of the tensor defining the positions to encode with an
   * 'on' value.
   * */
  static InIndex Indices() { return 1; }

  TensorId toEncodeId() const { return inTensorId(ToEncode()); }
  TensorId indicesId() const { return inTensorId(Indices()); }

private:
  /**
   * This op has 1 output, which needs to be modelled in an alias graph. It is
   * an alias of the input being encoded.
   * */
  void growAliasMapper(MemoryAliasMapper &mam) const final {
    createAlias(mam, toEncodeId());
  }

  void runSim(ISimState &ss) const final {
    runReplicatedSim(ss.simTensorMap());
  }

  /**
   * The output tensor is an alias of the input being encoded.
   * */
  bool aliases(InIndex i, OutIndex) const final { return i == ToEncode(); }

  /**
   * The input being encoded gets modified by this op.
   * */
  bool modifies(InIndex i) const final { return aliases(i, 0); }

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  /**
   * One hot encoding involves computation, so this op is not an 'initializing
   * op' like, for example, view-changing ops are.
   * */
  bool isInitializingOp() const final { return false; }

  /**
   * What device does this op operator on? Use the input and output tensors to
   * determine this. If they don't all agree, an error is thrown.
   * */
  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  /**
   * These methods are only pertinent to the RootRef_ op, whose output is an
   * alias of a tensor in a different sub-graph. For EncodeOneHot_, we use the
   * default behaviour:
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }

  HostTensors initializeOut(const HostTensors &) const final;
};

/**
 * One hot encoding with an 'on' value of 1 and an 'off' value of 0. Having
 * static 'on' and 'off' values allows for potentially faster implementations.
 * */
class EncodeOneHot01_ final
    : public Attributeless<EncodeOneHot_, EncodeOneHot01_> {

public:
  static constexpr const char *OpTypeName = "EncodeOneHot01_";
  EncodeOneHot01_(const State &s) : Attributeless(s) {}

private:
  void computeDerivedVerifyValid() const final;

  void compute(const HostTensors &, const HostTensors &) const final;
};

/**
 * One hot encoding where the on and off values not known at compile time.
 * They are provided as additional inputs to the op (meaning that this op has
 * 4 inputs).
 * */
class EncodeOneHotOffOn_ final
    : public Attributeless<EncodeOneHot_, EncodeOneHotOffOn_> {

public:
  static constexpr const char *OpTypeName = "EncodeOneHotOffOn_";
  EncodeOneHotOffOn_(const State &s) : Attributeless(s) {}

  static InIndex Off() { return 2; }
  static InIndex On() { return 3; }

private:
  void computeDerivedVerifyValid() const final;

  void compute(const HostTensors &, const HostTensors &) const final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
