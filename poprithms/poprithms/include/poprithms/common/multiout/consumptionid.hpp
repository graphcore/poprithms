// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_CONSUMPTIONID_HPP
#define POPRITHMS_COMMON_MULTIOUT_CONSUMPTIONID_HPP

#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * Description of an Op which consumes a Tensor. The class consists of the
 * consuming Op's OpId, and the input index at which the Tensor is consumed.
 * */
class ConsumptionId {

public:
  ConsumptionId() = delete;
  ConsumptionId(OpId opId__, InIndex inIndex__)
      : opId_(opId__), inIndex_(inIndex__) {}

  /** The OpId of the consuming Op */
  OpId opId() const { return opId_; }

  /** The InIndex at which the Tensor is consumed by the consuming Op */
  InIndex inIndex() const { return inIndex_; }

  void append(std::ostream &) const;
  std::string str() const;

  bool operator==(const ConsumptionId &rhs) const {
    return tup() == rhs.tup();
  }
  bool operator!=(const ConsumptionId &rhs) const {
    return tup() != rhs.tup();
  }
  bool operator<(const ConsumptionId &rhs) const { return tup() < rhs.tup(); }
  bool operator>(const ConsumptionId &rhs) const { return tup() > rhs.tup(); }
  bool operator<=(const ConsumptionId &rhs) const {
    return tup() <= rhs.tup();
  }
  bool operator>=(const ConsumptionId &rhs) const {
    return tup() >= rhs.tup();
  }

  std::tuple<OpId, InIndex> tup() const { return {opId(), inIndex()}; }

private:
  OpId opId_;
  InIndex inIndex_;
};

using ConsumptionIds = std::vector<ConsumptionId>;

std::ostream &operator<<(std::ostream &, const ConsumptionId &);
std::ostream &operator<<(std::ostream &, const ConsumptionIds &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
