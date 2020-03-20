#include <poprithms/schedule/anneal/allocweight.hpp>
#include <poprithms/schedule/anneal/trackentry.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

TrackEntry::TrackEntry(ScheduleIndex a, AllocWeight b, AllocWeight c, bool d)
    : entryTime(a), entryWeight(b), incrWeight(c), live(d) {}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
