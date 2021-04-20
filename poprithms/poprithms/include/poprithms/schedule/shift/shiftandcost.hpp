// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SHIFTANDCOST_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SHIFTANDCOST_HPP

#include <ostream>

#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class ShiftAndCost {
public:
  ShiftAndCost(ScheduleIndex i, AllocWeight c) : shift(i), cost(c) {}

  void append(std::ostream &ost) const;

  AllocWeight getCost() const { return cost; }
  ScheduleIndex getShift() const { return shift; }

  bool operator==(const ShiftAndCost &rhs) const {
    return shift == rhs.shift && cost == rhs.cost;
  }

  bool operator!=(const ShiftAndCost &rhs) const {
    return !(operator==(rhs));
  }

private:
  ScheduleIndex shift;
  AllocWeight cost;
};

std::ostream &operator<<(std::ostream &ost, const ShiftAndCost &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
