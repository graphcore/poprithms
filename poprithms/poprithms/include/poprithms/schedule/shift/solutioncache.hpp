// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SOLUTIONCACHE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SOLUTIONCACHE_HPP

#include <array>
#include <mutex>
#include <unordered_map>

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/schedulecache.hpp>
#include <poprithms/schedule/shift/settings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * \deprecated { This entire class is deprecated. Please use ScheduleCache
 *               instead. Use 'findExactStart' instead of 'find'. }
 * */
class SolutionCache : public ScheduleCache {
public:
  const std::vector<OpAddress> *find(const Graph &g,
                                     const Settings &s) const {
    auto x = findExactStart(g, s.rotationTermination());
    if (x.first) {
      exactSolnDeprecationSupport = x.second;
      return &exactSolnDeprecationSupport;
    } else {
      return nullptr;
    }
  }

private:
  /**
   * Using a mutable 'hack' for the period that this class is deprecated.
   * */
  mutable std::vector<OpAddress> exactSolnDeprecationSupport;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif

// #endif
