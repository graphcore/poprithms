#ifndef POPRITHMS_SCHEDULE_ANNEAL_TRACKENTRY_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_TRACKENTRY_HPP

#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

class TrackEntry {
public:
  TrackEntry(ScheduleIndex a, AllocWeight b, AllocWeight c, bool d)
      : entryTime(a), entryWeight(b), incrWeight(c), live(d) {}

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
