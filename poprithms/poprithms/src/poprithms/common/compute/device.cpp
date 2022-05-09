// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/device.hpp>

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

} // namespace compute
} // namespace common
} // namespace poprithms
