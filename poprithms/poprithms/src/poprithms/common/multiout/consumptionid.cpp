// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

void ConsumptionId::append(std::ostream &ost) const {
  ost << "(op=" << opId() << ",in=" << inIndex() << ')';
}

std::string ConsumptionId::str() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

std::ostream &operator<<(std::ostream &ost, const ConsumptionId &c) {
  c.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost,
                         const ConsumptionIds &consumptionIds) {
  util::append(ost, consumptionIds);
  return ost;
}

} // namespace multiout
} // namespace common
} // namespace poprithms
