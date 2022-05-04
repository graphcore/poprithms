// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <ostream>

#include <poprithms/ndarray/tensorinfo.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace ndarray {

TensorInfo TensorInfo::withShape(const Shape &s) const & {
  return {s, deviceId(), dtype()};
}

TensorInfo TensorInfo::withShape(const Shape &s) && {
  shape_ = s;
  return *this;
}

TensorInfo TensorInfo::withDeviceId(DeviceId id) const & {
  return {shape(), id, dtype()};
}

TensorInfo TensorInfo::withDeviceId(DeviceId id) && {
  deviceId_ = id;
  return *this;
}

TensorInfo TensorInfo::withDType(DType type) const & {
  return {shape(), deviceId(), type};
}

TensorInfo TensorInfo::withDType(DType type) && {
  dtype_ = type;
  return *this;
}

std::vector<Shape> TensorInfos::shapes() const {
  std::vector<Shape> shapes;
  shapes.reserve(size());
  for (const auto &ti : tensorInfos_) {
    shapes.push_back(ti.shape());
  }
  return shapes;
}

std::vector<DeviceId> TensorInfos::deviceIds() const {
  std::vector<DeviceId> dps;
  dps.reserve(size());
  for (const auto &ti : tensorInfos_) {
    dps.push_back(ti.deviceId());
  }
  return dps;
}

std::vector<DType> TensorInfos::dtypes() const {
  std::vector<DType> types;
  types.reserve(size());
  for (const auto &ti : tensorInfos_) {
    types.push_back(ti.dtype());
  }
  return types;
}

void TensorInfo::append(std::ostream &ost) const {
  ost << "Shape=" << shape() << ", DeviceId=" << deviceId()
      << ", DType=" << dtype();
}

void TensorInfos::append(std::ostream &ost) const {
  poprithms::util::append(ost, tensorInfos_);
}

std::ostream &operator<<(std::ostream &ost, const TensorInfo &tInfo) {
  tInfo.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const TensorInfos &tInfos) {
  tInfos.append(ost);
  return ost;
}

//
// DeviceId methods here, to avoid tiny translation units:
//
std::ostream &operator<<(std::ostream &ost, const DeviceIds &ids) {
  poprithms::util::append(ost, ids);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const DeviceId &id) {
  ost << id.get_u32();
  return ost;
}

} // namespace ndarray
} // namespace poprithms
