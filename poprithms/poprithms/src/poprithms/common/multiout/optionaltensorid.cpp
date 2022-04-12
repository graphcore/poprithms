// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <common/multiout/error.hpp>

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

std::vector<OptionalTensorId>
OptionalTensorId::fromTensorIds(const TensorIds &tIds) {
  std::vector<OptionalTensorId> otIds;
  otIds.reserve(tIds.size());
  for (const auto &tId : tIds) {
    otIds.push_back(tId);
  }
  return otIds;
}

} // namespace multiout
} // namespace common
} // namespace poprithms
