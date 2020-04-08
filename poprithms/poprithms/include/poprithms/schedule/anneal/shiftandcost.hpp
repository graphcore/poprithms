#ifndef POPRITHMS_SCHEDULE_ANNEAL_SHIFTANDCOST_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_SHIFTANDCOST_HPP

#include <poprithms/schedule/anneal/allocweight.hpp>
#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

class ShiftAndCost {
public:
  ShiftAndCost(ScheduleIndex i, AllocWeight c) : shift(i), cost(c) {}

  void append(std::ostream &ost) const {
    ost << "shift:" << shift << " cost:" << cost;
  }

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

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
