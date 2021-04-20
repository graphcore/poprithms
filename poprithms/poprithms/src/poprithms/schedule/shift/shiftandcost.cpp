// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <array>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/kahntiebreaker.hpp>
#include <poprithms/schedule/shift/shiftandcost.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

void ShiftAndCost::append(std::ostream &ost) const {
  ost << "shift:" << shift << " cost:" << cost;
}

std::ostream &operator<<(std::ostream &ost, const ShiftAndCost &x) {
  x.append(ost);
  return ost;
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
