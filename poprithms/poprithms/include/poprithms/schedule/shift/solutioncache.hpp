// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SOLUTIONCACHE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SOLUTIONCACHE_HPP

#include <array>
#include <mutex>
#include <unordered_map>

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/settings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/** Abstract base class for reading and writing solutions (schedules) to a
 * cache. */
class ISolutionCache {

public:
  /**
   * Return the solution in the cache fot the Graph #g and the Settings #s. If
   * there is no cached solution, nullptr is returned.
   * */
  virtual const std::vector<OpAddress> *find(const Graph &g,
                                             const Settings &s) const = 0;

  /**
   * Write the solution #soln for the Graph #g, scheduled with settings #s.
   * */
  virtual void writeSolution(Graph &&g,
                             const Settings &s,
                             const std::vector<OpAddress> &soln) = 0;
};

class SolutionCache : public ISolutionCache {
public:
  const std::vector<OpAddress> *find(const Graph &,
                                     const Settings &) const final;

  void writeSolution(Graph &&,
                     const Settings &,
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
  std::unordered_map<Graph,
                     std::map<Settings, std::vector<OpAddress>>,
                     IgnoreNamesHash,
                     IgnoreNamesEquals>
      cache;

  std::mutex mut;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
