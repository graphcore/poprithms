// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_DEVICETYPE_HPP
#define POPRITHMS_COMMON_COMPUTE_DEVICETYPE_HPP

#include <ostream>
#include <vector>

namespace poprithms {
namespace common {
namespace compute {

/**
 * The basic device types in a standard Graphcore system.
 * */
enum class DeviceType { Host = 0, Ipu, Remote };

using DeviceTypes = std::vector<DeviceType>;

std::ostream &operator<<(std::ostream &, DeviceType);
std::ostream &operator<<(std::ostream &, const DeviceTypes &);

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
