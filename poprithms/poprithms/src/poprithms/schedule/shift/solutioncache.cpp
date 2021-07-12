// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <mutex>
#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/solutioncache.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

const std::vector<OpAddress> *
SolutionCache::find(const Graph &graph, const Settings &settings) const {

  const auto found0 = cache.find(graph);
  if (found0 != cache.cend()) {
    const auto found1 = found0->second.find(settings);
    if (found1 != found0->second.cend()) {
      return &found1->second;
    }
  }

  return nullptr;
}

void SolutionCache::writeSolution(Graph &&graph,
                                  const Settings &settings,
                                  const std::vector<OpAddress> &soln) {

  const std::lock_guard<std::mutex> lock(mut);

  auto found0 = cache.find(graph);
  if (found0 == cache.cend()) {
    auto x = cache.insert({std::move(graph), {}});
    found0 = x.first;
  }

  if (found0->second.find(settings) != found0->second.cend()) {
    throw error("Conservatively throwing error: attempt to write "
                "ScheduledGraph Solution in cache which already has an entry "
                "for this Graph and Settings. ");
  }

  found0->second.insert({settings, soln});
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
