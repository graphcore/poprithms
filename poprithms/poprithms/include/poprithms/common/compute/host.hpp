// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_HOST_HPP
#define POPRITHMS_COMMON_COMPUTE_HOST_HPP

#include <poprithms/common/compute/device.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * The host device type.
 * */
class Host : public Device {
public:
  Host(DeviceId id) : Device(id, DeviceType::Host) {}
  std::unique_ptr<Device> clone() const final;

private:
  /**
   * There is no restriction on the shape or type of host tensor, so these
   * methods return true.
   * */
  bool canStoreShape(const Shape &) const final { return true; }
  bool canStoreDType(DType) const final { return true; }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
