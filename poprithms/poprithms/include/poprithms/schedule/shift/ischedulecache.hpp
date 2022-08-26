// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_ISCHEDULECACHE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_ISCHEDULECACHE_HPP

#include <unordered_map>
#include <vector>

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/rotationtermination.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class ScheduledGraph;

/**
 * Abstract base class for reading and writing solutions (schedules).
 * */
class IScheduleCache {

public:
  virtual ~IScheduleCache();
  IScheduleCache();

  virtual std::pair<bool, std::vector<OpAddress>>
  findExactStart(const Graph &g, const RotationTermination &r) const = 0;

  virtual void writeExactStart(Graph &&g,
                               const RotationTermination &rt,
                               const std::vector<OpAddress> &soln) = 0;

private:
  virtual void noWeakVTables();
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
