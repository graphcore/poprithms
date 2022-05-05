// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

OpId Graph::insertComputeOp(std::unique_ptr<Op> op) {
  return insertSchedulableOp(std::move(op));
}

DType Graph::dtype(const TensorId &tId) const {
  auto t = computeOp(tId.opId()).outDType(tId.outIndex());
  return t;
}

const Op &Graph::computeOp(OpId a) const {
  // We know that all Ops in this Graph can be safely cast, so no need for
  // dynamic_cast here.
  return static_cast<const Op &>(multioutOp(a));
}

const Op &Graph::op(OpId a) const { return computeOp(a); }

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
}

DeviceIds Graph::deviceIds(const TensorIds &tIds) const {
  DeviceIds devIds;
  for (auto tId : tIds) {
    devIds.push_back(deviceId(tId));
  }
  return devIds;
}

DeviceId Graph::deviceIdByConsensus(const TensorIds &all) const {

  if (all.empty()) {
    std::ostringstream oss;
    oss << "Unable to determine a DeviceId from empty set of tensor ids.";
    throw error(oss.str());
  }

  const auto dIds = deviceIds(all);

  for (auto devId : dIds) {
    if (devId != dIds[0]) {
      std::ostringstream oss;
      oss << "Failed to determine a DeviceId by consensus "
          << ", as not all tensors have the same DeviceId. "
          << "The DeviceIds were " << dIds << ", corresponding to tensors "
          << all << ", which are in sub-graphs " << subGraphIds(all) << '.';
      throw error(oss.str());
    }
  }

  return dIds[0];
}

bool Graph::isFixedPoint(const TensorId &tId) const {
  return poprithms::ndarray::isFixedPoint(dtype(tId));
}

DeviceId Graph::deviceId(const TensorId &tId) const {
  return op(tId.opId()).outDeviceId(tId.outIndex());
}

DeviceIds Graph::inDeviceIds(OpId opId) const {
  return op(opId).inDeviceIds();
}

DeviceIds Graph::outDeviceIds(OpId opId) const {
  return op(opId).outDeviceIds();
}
TensorInfo Graph::tensorInfo(const TensorId &tId) const {
  return op(tId.opId()).outTensorInfo(tId.outIndex());
}

TensorInfos Graph::tensorInfos(const TensorIds &ids) const {
  std::vector<TensorInfo> infos;
  infos.reserve(ids.size());
  for (const auto &id : ids) {
    infos.push_back(tensorInfo(id));
  }
  return infos;
}

} // namespace compute
} // namespace common
} // namespace poprithms
