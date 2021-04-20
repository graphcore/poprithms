// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SCHEDULECHANGE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SCHEDULECHANGE_HPP

#include <ostream>

#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class ScheduleChange {
public:
  ScheduleChange(ScheduleIndex s0, ScheduleIndex s1, int nts)
      : start0(s0), start1(s1), nToShift(nts) {}

  void append(std::ostream &ost) const {
    ost << "start0:" << start0 << " start1:" << start1
        << " nToShift:" << nToShift;
  }

  ScheduleChange getCanonical() const {
    if (start0 < start1) {
      return *this;
    }
    return {start1, start1 + nToShift, start0 - start1};
  }

  ScheduleIndex getStart0() const { return start0; }
  ScheduleIndex getStart1() const { return start1; }
  int getNToShift() const { return nToShift; }

private:
  // before the change
  ScheduleIndex start0;
  // after the change
  ScheduleIndex start1;
  // elements will move [start0, start0 + nToShift) ->
  //                    [start1, start1 + nToShift)
  int nToShift;
};

std::ostream &operator<<(std::ostream &ost, const ScheduleChange &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
