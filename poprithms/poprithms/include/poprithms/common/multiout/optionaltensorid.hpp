// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_OPTIONALTENSORID_HPP
#define POPRITHMS_COMMON_MULTIOUT_OPTIONALTENSORID_HPP

#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace common {
namespace multiout {
/**
 * This class behaves like std::optional<TensorId>, but without requiring
 * c++17.
 * */
class OptionalTensorId {
public:
  OptionalTensorId() : isSet(false) {}
  OptionalTensorId(const TensorId &id_) : id(id_), isSet(true) {}
  const TensorId &value() const;
  bool has_value() const { return isSet; }
  void append(std::ostream &) const;

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
