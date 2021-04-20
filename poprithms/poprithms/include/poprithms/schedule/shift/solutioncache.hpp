// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SOLUTIONCACHE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SOLUTIONCACHE_HPP

#include <array>
#include <map>
#include <mutex>

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/settings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class SolutionCache {
public:
  const std::vector<OpAddress> *find(const Graph &, const Settings &) const;

  void
  writeSolution(Graph &&, const Settings &, const std::vector<OpAddress> &);

private:
  // when comparing Graphs, ignore the Op names.
  struct Comparitor {
    bool operator()(const Graph &a, const Graph &b) const {
      return a.lessThan(b, false);
    }
  };
  std::map<Graph, std::map<Settings, std::vector<OpAddress>>, Comparitor>
      cache;

  std::mutex mut;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
