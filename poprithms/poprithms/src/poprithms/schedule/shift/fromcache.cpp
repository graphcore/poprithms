// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/schedule/shift/fromcache.hpp>
#include <poprithms/schedule/shift/logging.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

std::tuple<CacheHitType, std::vector<OpAddress>>
probeCache(const Graph &graph,
           const RotationTermination &rt,
           const IScheduleCache *cache) {

  if (!cache) {
    return {CacheHitType::Cold, {}};
  }

  auto hotFind = cache->findExactStart(graph, rt);
  if (hotFind.first) {
    return {CacheHitType::Hot, hotFind.second};
  }

  return {CacheHitType::Cold, {}};
}

std::ostream &operator<<(std::ostream &ost, CacheHitType chit) {
  switch (chit) {
  case CacheHitType::Hot: {
    ost << "Hot";
    break;
  }
  case CacheHitType::Cold: {
    ost << "Cold";
    break;
  }
  }
  return ost;
}

ScheduledGraph fromCache(Graph &&graph,
                         const Settings &inputSettings,
                         const ISummaryWriter &summaryWriter,
                         const IScheduleCache *readCache,
                         IScheduleCache *writeCache) {

  const auto cacheFind =
      probeCache(graph, inputSettings.rotationTermination(), readCache);

  switch (std::get<0>(cacheFind)) {
  case CacheHitType::Hot: {
    const auto &soln = std::get<1>(cacheFind);
    for (uint64_t i = 1; i < soln.size(); ++i) {
      graph.insertConstraint(soln[i - 1], soln[i]);
    }
    auto hotSettings = Settings(KahnDecider(KahnTieBreaker::FIFO),
                                TransitiveClosureOptimizations::allOff(),
                                RotationTermination::preStart());
    return ScheduledGraph(std::move(graph), hotSettings, FileWriter::None());
  }

  case CacheHitType::Cold: {
    break;
  }
  }

  if (!writeCache) {
    return ScheduledGraph(std::move(graph), inputSettings, summaryWriter);
  } else {

    // we need a copy of the user's graph, as this will be the key in the
    // (hot) cache. When we call initialize, the graph might change to make it
    // easier to schedule (using transitive closure optimizations). However,
    // the key we want to cache is the original user's graph.
    auto g0 = graph;

    auto soln =
        ScheduledGraph(std::move(graph), inputSettings, summaryWriter);
    std::vector<OpAddress> inputGraphAdds(g0.nOps());
    std::iota(inputGraphAdds.begin(), inputGraphAdds.end(), 0);
    auto subSchedule = soln.getSubSchedule(inputGraphAdds);
    writeCache->writeExactStart(
        std::move(g0), inputSettings.rotationTermination(), subSchedule);

    return soln;
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
