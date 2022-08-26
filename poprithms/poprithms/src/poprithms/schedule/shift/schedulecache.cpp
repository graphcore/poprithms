// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <mutex>
#include <unordered_map>

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/schedulecache.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

void IScheduleCache::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

IScheduleCache::~IScheduleCache() = default;

IScheduleCache::IScheduleCache() = default;

std::pair<bool, std::vector<OpAddress>>
ScheduleCache::findExactStart(const Graph &graph,
                              const RotationTermination &rt) const {

  const auto found0 = exactStarts.find(graph);
  if (found0 != exactStarts.cend()) {
    for (const auto &soln : found0->second) {
      if (soln.first == rt) {
        return {true, soln.second};
      }
    }
  }
  return {false, {}};
}

void ScheduleCache::writeExactStart(Graph &&graph,
                                    const RotationTermination &rt,
                                    const std::vector<OpAddress> &soln) {

  const std::lock_guard<std::mutex> lock(mut);

  auto found0 = exactStarts.find(graph);
  if (found0 == exactStarts.cend()) {
    auto x = exactStarts.insert({std::move(graph), {}});
    found0 = x.first;
  }

  for (const auto &x : found0->second) {
    if (x.first == rt) {
      std::ostringstream oss;
      oss << "Attemp to write "
             "ScheduledGraph solution of size "
          << soln.size()
          << " in this ScheduleCache, which already has an entry "
          << "for this Graph and RotationTermination."
          << " Assuming that this is an error and bailing. ";
      throw error(oss.str());
    }
  }

  found0->second.push_back({rt, soln});
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
