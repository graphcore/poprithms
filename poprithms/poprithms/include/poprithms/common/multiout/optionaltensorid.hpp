// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_OPTIONALTENSORID_HPP
#define POPRITHMS_COMMON_MULTIOUT_OPTIONALTENSORID_HPP

#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * This class provided a subset of the functionality of
 * std::optional<TensorId>, without requiring c++17.
 * */
class OptionalTensorId {
public:
  OptionalTensorId() : isSet(false) {}
  OptionalTensorId(const TensorId &id_) : id(id_), isSet(true) {}
  const TensorId &value() const;
  bool has_value() const { return isSet; }
  void append(std::ostream &) const;

  bool operator==(const OptionalTensorId &rhs) const {
    if (has_value() != rhs.has_value()) {
      return false;
    }
    if (rhs.has_value() && id != rhs.id) {
      return false;
    }
    // neither has a value.
    return true;
  }

  bool operator<(const OptionalTensorId &rhs) const {

    // neither has a value: they're equal.
    if (!has_value() && !rhs.has_value()) {
      return false;
    }

    // both have a value: compare values.
    if (has_value() && rhs.has_value()) {
      return id < rhs.id;
    }

    // no-value is always less than value.
    if (!has_value()) {
      return true;
    }

    // value is always greater than no-value.
    return false;
  }

  bool operator!=(const OptionalTensorId &rhs) const {
    return !operator==(rhs);
  }
  bool operator>=(const OptionalTensorId &rhs) const {
    return !operator<(rhs);
  }
  bool operator>(const OptionalTensorId &rhs) const {
    return !operator==(rhs) && !operator<(rhs);
  }
  bool operator<=(const OptionalTensorId &rhs) const {
    return !operator>(rhs);
  }

private:
  TensorId id;
  bool isSet{false};
};

using OptionalTensorIds = std::vector<OptionalTensorId>;

std::ostream &operator<<(std::ostream &, const OptionalTensorId &);
std::ostream &operator<<(std::ostream &, const OptionalTensorIds &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
