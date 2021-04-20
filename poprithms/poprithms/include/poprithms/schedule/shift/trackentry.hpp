// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_TRACKENTRY_HPP
#define POPRITHMS_SCHEDULE_SHIFT_TRACKENTRY_HPP

#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class TrackEntry {
public:
  TrackEntry(ScheduleIndex a, AllocWeight b, AllocWeight c, bool d)
      : entryTime(a), entryWeight(b), incrWeight(c), live(d) {}

  TrackEntry(const TrackEntry &) = default;
  TrackEntry(TrackEntry &&)      = default;

  TrackEntry &operator=(const TrackEntry &) = default;
  TrackEntry &operator=(TrackEntry &&) = default;

  // when registered
  ScheduleIndex entryTime;

  // cost when registered
  AllocWeight entryWeight;

  // amount to increment cumulative cost at each iteration
  AllocWeight incrWeight;

  bool live;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
