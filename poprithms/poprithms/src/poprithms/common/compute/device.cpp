// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/device.hpp>
#include <poprithms/common/compute/host.hpp>
#include <poprithms/common/compute/ipu.hpp>
#include <poprithms/common/compute/remote.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace common {
namespace compute {

std::string Device::str() const {
  std::ostringstream oss;
  oss << deviceType_ << "(id=" << id() << ")";
  return oss.str();
}

std::ostream &operator<<(std::ostream &oss, const Device &d) {
  oss << d.str();
  return oss;
}

void Device::confirmCanStore(const Shape &s, DType d) const {
  if (!canStoreShape(s)) {
    std::ostringstream oss;
    oss << *this << " cannot store a tensor of shape " << s;
    throw error(oss.str());
  }
  if (!canStoreDType(d)) {
    std::ostringstream oss;
    oss << *this << " cannot store a tensor of type "
        << poprithms::ndarray::lcase(d);
    throw error(oss.str());
  }
}

std::unique_ptr<Device> Host::clone() const {
  return std::make_unique<Host>(id());
}

using Interval  = poprithms::util::Interval;
using Intervals = poprithms::util::Intervals;

bool Ipu::canStoreDType(DType d) const {
  switch (d) {

  // Ipus can store these:
  case DType::Float32:
  case DType::Float16:
  case DType::Unsigned32:
  case DType::Int32:
  case DType::Unsigned16:
  case DType::Int16:
  case DType::Unsigned8:
  case DType::Int8:
  case DType::Boolean: {
    return true;
  }

  // Ipus can't store 64-bit types:
  case DType::Float64:
  case DType::Int64:
  case DType::Unsigned64: {
    return false;
  }
  case DType::N: {
    throw error("DType::N is not an actual type");
  }
  }
  throw error("Unprocessed DType in Ipu::canStoreDType");
}

std::unique_ptr<Device> Ipu::clone() const {
  return std::make_unique<Ipu>(id(), tiles());
}

bool Remote::canStoreShape(const Shape &s) const { return s.rank_u64() == 2; }

std::unique_ptr<Device> Remote::clone() const {
  return std::make_unique<Remote>(id(), ipu_, dtype_, shape_, options_);
}

Remote::Remote(DeviceId remoteId,
               DeviceId ipuId,
               DType type,
               const Shape &shape,
               const RemoteOptions &ros)
    : Device(remoteId, DeviceType::Remote), ipu_(ipuId), dtype_(type),
      shape_(shape), options_(ros) {

  if (shape.rank_u64() != 2) {
    throw error("Remote device: shape must be rank-2");
  }
}

bool Remote::canStoreDType(DType) const { return true; }

RemoteOptions &RemoteOptions::rearrangeOnHost(bool b) {
  rearrangeOnHost_ = b;
  return *this;
}

RemoteOptions &RemoteOptions::optimizeMemory(bool b) {
  optimizeMemory_ = b;
  return *this;
}

RemoteOptions &RemoteOptions::handle(const std::string &h) {
  handle_ = h;
  return *this;
}

void Device::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

} // namespace compute
} // namespace common
} // namespace poprithms
