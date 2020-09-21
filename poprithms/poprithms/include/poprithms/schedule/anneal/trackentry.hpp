// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_ANNEAL_TRACKENTRY_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_TRACKENTRY_HPP

#include <poprithms/schedule/anneal/allocweight.hpp>
#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

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

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
