// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SCHEDULECACHE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SCHEDULECACHE_HPP

#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <poprithms/schedule/shift/ischedulecache.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class ScheduleCache : public IScheduleCache {
public:
  /**
   * Return the solution in the cache for the Graph #g. Only solutions which
   * were obtained with the RotationTermination #r are considered. If there is
   * no cached solution found, the returned pair has its first value as
   * 'false'.
   * */
  virtual std::pair<bool, std::vector<OpAddress>>
  findExactStart(const Graph &, const RotationTermination &) const final;

  /**
   * Write the solution #soln for the Graph #g, scheduled with
   * RotationTermination #rt.
   * */
  void writeExactStart(Graph &&,
                       const RotationTermination &,
                       const std::vector<OpAddress> &) final;

private:
  // when comparing Graphs in the cache, ignore the Op names.
  struct IgnoreNamesHash {
    size_t operator()(const Graph &a) const { return a.hash(false); }
  };
  struct IgnoreNamesEquals {
    bool operator()(const Graph &a, const Graph &b) const {
      return a.equalTo(b, false);
    }
  };

  std::unordered_map<
      Graph,
      std::vector<std::pair<RotationTermination, std::vector<OpAddress>>>,
      IgnoreNamesHash,
      IgnoreNamesEquals>
      exactStarts;

  std::mutex mut;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
