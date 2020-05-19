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
  TrackEntry(ScheduleIndex, AllocWeight, AllocWeight, bool);

  ScheduleIndex entryTime; // when registered
  AllocWeight entryWeight; // cost when registered
  AllocWeight
      incrWeight; // amount to increment cumulative cost at each iteration
  bool live;
};

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
