// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iterator>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/init.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Init
 * */
void Init::invalidInIndex(InIndex i) const {
  std::ostringstream oss;
  oss << "Invalid input index " << i << " for Init of " << *this
      << " : Init ops have no inputs. ";
  throw error(oss.str());
}

/**
 * ConstInit
 * */
ConstInit::ConstInit(const State &s,
                     const poprithms::compute::host::Tensor &val)
    : poprithms::common::compute::Init(s), value_(val) {
  // Note that the shape and type of val is checked in
  // computeDerivedVerifyValid. This design pattern, of running checks of an
  // ops correctness in the virtual method computeDerivedVerifyValid, must be
  // followed by all ops. It allows easy checking of op correctness when an op
  // is modified.
}

void ConstInit::growAliasMapper(MemoryAliasMapper &b) const {
  b.insert({b.graph().allocate(outShape(0), MemoryAliasConstant)},
           outTensorIds());
}

std::string ConstInit::typeString() const {
  std::ostringstream oss;
  oss << "ConstInit";
  if (value_.nelms_u64() == 1) {
    oss << "(" << value_.valueAsStr(0) << ")";
  }
  return oss.str();
}

bool ConstInit::computeTypeSpecificEqualTo(const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const ConstInit &>(rhs);
  return value().numericallyIdenticalTo(rhs_.value());
}

UpOp ConstInit::cloneConstInitWithState(const State &s,
                                        bool pointerOnly) const {

  auto clonedValue = pointerOnly ? value() : value().copy();
  return std::make_unique<ConstInit>(s, clonedValue);
}

void ConstInit::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(0, 1, {});
  if (value_.shape() != outShape(0)) {
    std::ostringstream oss;
    oss << "Op " << *this << " has output shape " << outShape(0)
        << " but the stored value of the constant has shape "
        << value_.shape() << ". They should be identical.";
    throw error(oss.str());
  }

  if (value_.dtype() != outDType(0)) {
    std::ostringstream oss;
    oss << "Op " << *this << " has output type " << outDType(0)
        << " but the stored value of the constant has type " << value_.dtype()
        << ". They should be the same.";
    throw error(oss.str());
  }

  computeGraph()
      .device(outDeviceId(0))
      .confirmCanStore(outShape(0), outDType(0));
}

/**
 * VarInit
 * */

std::string VarInit::typeString() const {
  return !isUserManagedHost() ? "VarInit" : "VarInit(user-managed-pointer)";
}

void VarInit::growAliasMapper(MemoryAliasMapper &b) const {
  return createVariables(b);
}

void VarInit::computeDerivedVerifyValid() const {
  poprithms::common::compute::OpVerifier(*this).verifyNonVariadicFromAtts(
      0, 1, {});

  if (isUserManagedHost() && outDeviceType(0) != DeviceType::Host) {
    throw error("Only host tensors can be user managed.");
  }
  computeGraph()
      .device(outDeviceId(0))
      .confirmCanStore(outShape(0), outDType(0));
}

bool VarInit::isUserManagedHost() const {
  return userManagedHost_ == UserManagedHost::Yes;
}

bool VarInit::computeTypeSpecificEqualTo(const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const VarInit &>(rhs);
  return userManagedHost_ == rhs_.userManagedHost_;
}

void VarInit::setUserManagedHost(bool b) {
  if (outDeviceType(0) != DeviceType::Host) {
    std::ostringstream oss;
    oss << "This VarInit op " << *this << " has device type "
        << outDeviceType(0)
        << " only host ops should have this attribute set.";
    throw error(oss.str());
  }
  userManagedHost_ = b ? UserManagedHost::Yes : UserManagedHost::No;
}

poprithms::compute::host::Tensors
VarInit::initializeOut(const poprithms::compute::host::Tensors &) const {
  if (isUserManagedHost()) {
    return {HostTensor::uninitializedRef(outShape(0), outDType(0))};
  } else {
    return badValOuts();
  }
}

UpOp VarInit::cloneWithState(const State &s) const {
  auto upBop = std::make_unique<VarInit>(s);
  if (isUserManagedHost()) {
    upBop->setUserManagedHost(true);
  }
  return upBop;
}

} // namespace compute
} // namespace common
} // namespace poprithms
