// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/opverifier.hpp>

namespace poprithms {
namespace common {
namespace compute {

void OpVerifier::verifyNonVariadicFromAtts(
    uint64_t nIn,
    uint64_t nOut,
    const std::vector<Att> &atts) const {
  op.verifyNInAndOutTensors(nIn, nOut);
  verifyFromAtts(atts);
}

void OpVerifier::verifySameTensorInfo(InIndex inIndex,
                                      OutIndex outIndex) const {
  if (op.inTensorInfo(inIndex) != op.outTensorInfo(outIndex)) {
    std::ostringstream oss;
    oss << "Expected the input at index " << inIndex
        << " and the output at index " << outIndex << " of op " << op
        << " to have the same tensor informations. But the input info is "
        << op.inTensorInfo(inIndex) << " and the output info is "
        << op.outTensorInfo(outIndex);
    throw error(oss.str());
  }
}

void OpVerifier::verifyInIsFixedPoint(InIndex i) const {
  if (!poprithms::ndarray::isFixedPoint(op.inDType(i))) {
    std::ostringstream oss;
    oss << "Failed to verify that the input to the op " << op
        << ", at input index #" << i << ", is a fixed point tensor. "
        << "It has tensor info "
        << op.computeGraph().tensorInfo(op.inTensorId(i)) << '.';
    throw error(oss.str());
  }
}

void OpVerifier::verifyInsSameDType() const {
  if (op.nInTensors() < 2) {
    return;
  }
  for (uint64_t i = 1; i < op.nInTensors(); ++i) {
    if (op.inDType(i) != op.inDType(0)) {
      std::ostringstream oss;
      oss << "Failure in verifyInsSameDType for op " << op
          << ". The input #0 has dtype " << op.inDType(0)
          << ", but the input #" << i << " has dtype " << op.inDType(i)
          << '.';
      throw error(oss.str());
    }
  }
}

void OpVerifier::verifyOutsSameDType() const {
  if (op.nOutTensors() < 2) {
    return;
  }
  for (uint64_t i = 1; i < op.nOutTensors(); ++i) {
    if (op.outDType(i) != op.outDType(0)) {
      std::ostringstream oss;
      oss << "Failure in OpVerifier::verifyOutsSameDType for op " << op
          << ". The output #0 has dtype " << op.outDType(0)
          << ", but the output #" << i << " has dtype " << op.outDType(i)
          << '.';
      throw error(oss.str());
    }
  }
}

void OpVerifier::verifyAllSameDType() const {

  verifyInsSameDType();
  verifyOutsSameDType();

  if (op.nInTensors() > 0 && op.nOutTensors() > 0) {
    if (op.outDType(0) != op.inDType(0)) {
      std::ostringstream oss;
      oss << "Failure in OpVerifier::verifyAllSameDType for op " << op
          << ". The output #0 has dtype " << op.outDType(0)
          << ", but the input #0"
          << " has dtype " << op.inDType(0) << '.';
      throw error(oss.str());
    }
  }
}

void OpVerifier::verifyFromAtts(const std::vector<Att> &atts) const {

  verifyDeviceCompatibilityOfOutputs();
  for (auto att : atts) {
    switch (att) {

    case Att::SameDType: {
      verifyAllSameDType();
      break;
    }
    case Att::InsSameDType: {
      verifyInsSameDType();
      break;
    }
    case Att::FloatIfIpu: {
      verifyAllFloatingIfIpu();
      break;
    }
    case Att::SameDeviceType: {
      verifyAllSameDeviceType();
      break;
    }
    case Att::SameDevice: {
      verifyAllSameDevice();
      break;
    }
    }
  }
}

void OpVerifier::verifyAllSameDeviceType() const {
  auto outTypes = op.outDeviceTypes();
  auto allTypes = op.inDeviceTypes();
  allTypes.insert(allTypes.end(), outTypes.cbegin(), outTypes.cend());

  if (allTypes.empty()) {
    return;
  }

  if (std::any_of(allTypes.cbegin(),
                  allTypes.cend(),
                  [&allTypes](DeviceType a) { return a != allTypes[0]; })) {
    std::ostringstream oss;
    oss << "Failure in OpVerifier::verifyAllSameDeviceType for op " << op
        << ". The input device types are " << op.inDeviceTypes()
        << " and the output device types are " << op.outDeviceTypes() << '.';
    throw error(oss.str());
  }
}

void OpVerifier::verifyAllSameDevice() const {
  auto outIds = op.outDeviceIds();
  auto allIds = op.inDeviceIds();
  allIds.insert(allIds.end(), outIds.cbegin(), outIds.cend());

  if (allIds.empty()) {
    return;
  }

  if (std::any_of(allIds.cbegin(), allIds.cend(), [&allIds](DeviceId b) {
        return b != allIds[0];
      })) {
    std::ostringstream oss;
    oss << "Failure in OpVerifier::verifyAllSameDevice for op " << op
        << ". The input device ids are " << op.inDeviceIds()
        << " and the output device ids are " << op.outDeviceIds() << '.';
    throw error(oss.str());
  }
}

void OpVerifier::verifyAllIpu() const {

  auto getMessage = [this](Op::Port port, uint64_t index) {
    std::ostringstream oss;
    oss << "Failure in OpVerifier::verifyAllIpu for op " << op << ". The "
        << Op::lowercase(port) << "put tensor #" << index
        << " (tensor id = " << op.tensorId(port, index) << ") is on device "
        << op.device(port, index)
        << ", which is of device type: " << op.deviceType(port, index) << '.';
    return oss.str();
  };

  for (auto p : {Op::Port::In, Op::Port::Out}) {
    for (uint64_t i = 0; i < op.nTensors(p); ++i) {
      if (op.deviceType(p, i) != DeviceType::Ipu) {
        throw error(getMessage(p, i));
      }
    }
  }
}

void OpVerifier::verifyAllFloatingIfIpu() const {

  auto getMessage = [this](Op::Port port, uint64_t index) {
    std::ostringstream oss;
    oss << "Failure in OpVerifier::verifyFloatIfIpu for op " << op << ". The "
        << Op::lowercase(port) << "put Tensor #" << index
        << " (tensor id = " << op.tensorId(port, index) << ")  is of type "
        << op.dtype(port, index) << ", and is on device "
        << op.device(port, index) << '.';
    return oss.str();
  };

  for (auto p : {Op::Port::In, Op::Port::Out}) {
    for (uint64_t i = 0; i < op.nTensors(p); ++i) {
      if (op.deviceType(p, i) == DeviceType::Ipu &&
          poprithms::ndarray::isFixedPoint(op.dtype(p, i))) {
        throw error(getMessage(p, i));
      }
    }
  }
}

void OpVerifier::verifyDeviceCompatibilityOfOutputs() const {

  // Each op can check its own outputs.
  for (OutIndex o = 0; o < op.nOutTensors(); ++o) {

    auto getBase = [this, &o]() {
      std::ostringstream oss;
      oss << "Failure in OpVerifier::verifyDeviceCompatibility"
          << " for op " << op << ", and for output #" << o
          << ". The output has tensor info, " << op.outTensorInfo(o) << '.';
      return oss;
    };

    const auto &d = op.outDevice(o);

    if (!d.canStoreShape(op.outShape(o))) {
      auto oss = getBase();
      oss << "The tensor's shape is invalid for " << d << '.';
      throw error(oss.str());
    }

    if (!d.canStoreDType(op.outDType(o))) {
      auto oss = getBase();
      oss << "The tensor's data type (DType)"
          << " is invalid for " << d << '.';
      throw error(oss.str());
    }
  }
}

} // namespace compute
} // namespace common
} // namespace poprithms
