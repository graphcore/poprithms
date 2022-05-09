// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

std::ostream &operator<<(std::ostream &ost, DeviceType dt) {
  switch (dt) {
  case DeviceType::Remote: {
    ost << "Remote";
    break;
  }

  case DeviceType::Host: {
    ost << "Host";
    break;
  }

  case DeviceType::Ipu: {
    ost << "Ipu";
    break;
  }
  }

  return ost;
}

std::ostream &operator<<(std::ostream &oss, const DeviceTypes &dts) {
  poprithms::util::append(oss, dts);
  return oss;
}

} // namespace compute
} // namespace common
} // namespace poprithms
