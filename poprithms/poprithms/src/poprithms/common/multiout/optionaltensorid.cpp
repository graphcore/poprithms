// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/common/multiout/error.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

const TensorId &OptionalTensorId::value() const {
  if (!has_value()) {
    throw error(
        "Invalid call to OptionalTensorId::value(). has_value() is false.");
  }
  return id;
}

void OptionalTensorId::append(std::ostream &ost) const {
  if (has_value()) {
    ost << id;
  } else {
    ost << "none";
  }
}

std::ostream &operator<<(std::ostream &ost, const OptionalTensorId &ot) {
  ot.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const OptionalTensorIds &ids) {
  util::append(ost, ids);
  return ost;
}

} // namespace multiout
} // namespace common
} // namespace poprithms
