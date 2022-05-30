// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

HostTensors RefFrom::initializeOut(const HostTensors &) const {
  std::ostringstream oss;
  oss << "This code path should never be taken for ops of type RefFrom. "
      << "As this method only provides input tensors "
      << "(of which RefFrom has none) "
      << "there is not enough information. "
      << "RefFrom needs the more comprehensive tensor information of "
      << "initializeSimOut to have access to its root reference tensor. ";
  throw error(oss.str());
}

void RefFrom::resetRootRef(OutIndex o, const TensorId &root) {

  if (o != OutIndex(o)) {
    std::ostringstream oss;
    oss << "RootRef only has 1 output, invalid output index " << o
        << " for this op, " << *this;
    throw error(oss.str());
  }

  if (root == outTensorId(0)) {
    throw error("RefFrom must have a different root reference and output.");
  }

  root_ = root;
}

std::string RefFrom::typeString() const {
  return util::cat::strcat("RefFrom(", rootRef(0).str(), ')');
}

bool RefFrom::computeTypeSpecificEqualTo(const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const RefFrom &>(rhs);
  return root_ == rhs_.root_;
}

UpOp RefFrom::cloneWithState(const State &s) const {
  return std::make_unique<RefFrom>(s, rootRef(OutIndex(0)));
}

RefFrom::RefFrom(const State &s, const TensorId &root)
    : WithoutCallees(s), root_(root) {}

void RefFrom::initializeSimOut(SimTensorMap &htm) const {
  htm.setValue({id(), 0}, htm.getValue(rootRef(0)));
}

void RefFrom::growAliasMapper(MemoryAliasMapper &mam) const {
  // identity op, so that the new tensor in #mam has a different id to the
  // root ref's.
  createAlias(mam, rootRef(OutIndex(0)));
}

void RefFrom::computeDerivedVerifyValid() const {
  // 0 inputs, 1 output.
  OpVerifier(*this).verifyNonVariadicFromAtts(0, 1, {});
}

} // namespace compute
} // namespace common
} // namespace poprithms
