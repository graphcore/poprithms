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

void Graph::schedulableTypeSpecificRemoveOp(
    OpId opToRemove,
    const OptionalTensorIds &outputSubstitutes) {

  (void)opToRemove;
  (void)outputSubstitutes;
}

void Graph::schedulableTypeSpecificVerifyValidSubstitute(
    const TensorId &before,
    const TensorId &after) const {
  if (deviceId(before) != deviceId(after)) {
    std::ostringstream oss;
    oss << "DeviceId of tensor before substitution is " << deviceId(before)
        << ", and DeviceId after substitution is " << deviceId(after) << ". ";
    throw error(oss.str());
  }

  if (dtype(before) != dtype(after)) {
    std::ostringstream oss;
    oss << "Type of tensor before substitution is " << dtype(before)
        << ", and type after substitution is " << deviceId(after) << ". ";
    throw error(oss.str());
  }
}

DeviceType Graph::deviceType(const TensorId &tensorId) const {
  return device(tensorId).deviceType();
}

DeviceType Graph::deviceType(OpId opId) const {
  return op(opId).deviceType();
}
DeviceType Graph::deviceType(DeviceId devId) const {
  return device(devId).deviceType();
}

std::vector<DeviceType> Graph::deviceTypes(const TensorIds &ids) const {
  std::vector<DeviceType> ts;
  ts.reserve(ids.size());
  for (auto id : ids) {
    ts.push_back(deviceType(id));
  }
  return ts;
}

const Device &Graph::device(const TensorId &tid) const {
  return device(op(tid.opId()).outDeviceId(tid.outIndex()));
}

OpId Graph::insertComputeOp(std::unique_ptr<Op> nxtOp) {
  const auto newId = insertSchedulableOp(std::move(nxtOp));
  verifyValidAtComputeLevel(newId);
  return newId;
}

void Graph::verifyValidAtComputeLevel(OpId opId) const {
  op(opId).verifyValidAtComputeLevel();
}

void Graph::verifySchedulableDerivedGraphValid() const {
  for (auto opId : multiout::Graph::opIds()) {
    verifySchedulableDerivedOpValid(opId);
  }
}

void Graph::verifySchedulableDerivedOpValid(OpId opId) const {
  verifyValidAtComputeLevel(opId);
  verifyComputeDerivedOpValid(opId);
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
namespace {
template <typename T> std::string getStr(const std::vector<T> &X) {
  std::ostringstream ost;
  poprithms::util::append(ost, X);
  return ost.str();
}
} // namespace

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds_) const {

  const auto colParams = poprithms::util::StringColumn::Parameters()
                             .abridgeToSingleRow(false)
                             .thresholdWidth(50);

  auto cols = getMultioutColumns(opIds_, colParams);

  for (auto &&c : getSchedulableColumns(opIds_, colParams)) {
    cols.push_back(c);
  }

  for (auto &&c : getComputeColumns(opIds_, colParams)) {
    cols.push_back(c);
  }

  ost << alignedColumns(cols);
}

std::vector<poprithms::util::StringColumn> Graph::getComputeColumns(
    const OpIds &opIds_,
    const poprithms::util::StringColumn::Parameters &colParams) const {

  std::vector<poprithms::util::StringColumn> cols;

  const auto nRows = nMultioutRows(opIds_);
  using Strings    = std::vector<std::string>;

  Strings devices(nRows, "");
  Strings dtypes(nRows, "");
  Strings rootRefOf(nRows, "");
  bool hasOutRefs{false};

  uint64_t rowIndex{0};
  for (auto opId : opIds_) {

    const auto &op_       = op(opId);
    const auto subGraphId = op_.subGraphId();
    const auto gName      = subGraphName(subGraphId);
    for (uint64_t o = 0; o < op_.nOutTensors(); ++o) {
      dtypes[rowIndex]  = poprithms::ndarray::lcase(dtype({opId, o}));
      const auto devId  = op_.outDeviceId(o);
      devices[rowIndex] = device(devId).str();
      auto outRefs      = op_.derivedRefs(o);
      hasOutRefs |= !outRefs.empty();
      rootRefOf[rowIndex] = getStr(outRefs);
      ++rowIndex;
    }
    if (op_.nOutTensors() == 0) {
      ++rowIndex;
    }
  }

  cols.push_back({"Device", std::move(devices), colParams});

  // If there is no cross-graph referencing, do not append a column.
  if (hasOutRefs) {
    cols.push_back({"IsRootOf", std::move(rootRefOf), colParams});
  }
  cols.push_back({"Type", std::move(dtypes), colParams});

  return cols;
}

} // namespace compute
} // namespace common
} // namespace poprithms
