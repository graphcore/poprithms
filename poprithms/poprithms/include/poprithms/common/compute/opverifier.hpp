// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPVERIFIER_HPP
#define POPRITHMS_COMMON_COMPUTE_OPVERIFIER_HPP

#include <poprithms/common/compute/op.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * A class method for testing assumptions on ops.
 * */
class OpVerifier {

private:
  // The op which is being tested by this OpVerifier.
  const Op &op;

public:
  /**
   * Create a verifier for the op #op_
   * */
  OpVerifier(const Op &op_) : op(op_) {}

  /**
   * Verify that the types and shapes of the outputs of the op are compatible
   * with the devices they are on. See the Device class for more information.
   * */
  void verifyDeviceCompatibilityOfOutputs() const;

  /**
   * Verify that the op's inputs and outputs are on devices of the same type.
   * */
  void verifyAllSameDeviceType() const;

  /**
   * Verify that the op's inputs and outputs are on the same device.
   * */
  void verifyAllSameDevice() const;

  /**
   * Verify that all inputs and outputs which are on ipu are also floating
   * point tensors.
   * */
  void verifyAllFloatingIfIpu() const;

  /**
   * Verify that the op's input at index #i to is fixed point (integral).
   * */
  void verifyInIsFixedPoint(InIndex i) const;

  /**
   * Verify that all inputs have the same numerical type.
   * */
  void verifyInsSameDType() const;

  /**
   * Verify that all outputs have the same numerical type.
   * */
  void verifyOutsSameDType() const;

  /**
   * Verify that all inputs and outputs have the same numerical type.
   * */
  void verifyAllSameDType() const;

  enum class Att {
    /// All inputs and outputs are the same numerical type.
    SameDType = 0,
    /// All inputs are the same numerical type.
    InsSameDType,
    /// All inputs and outputs which are on ipu are floating point.
    FloatIfIpu,
    /// All inputs and outputs are on devices of the same type.
    SameDeviceType,
    /// All inputs and outputs are on the same device.
    SameDevice,
  };

  /**
   * Verify that the op has #nIns inputs and #nOuts outputs, and that all of
   * the attributes in #atts are satisfied.
   * */
  void verifyNonVariadicFromAtts(uint64_t nIns,
                                 uint64_t nOuts,
                                 const std::vector<Att> &atts) const;

  void verifyFromAtts(const std::vector<Att> &) const;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
