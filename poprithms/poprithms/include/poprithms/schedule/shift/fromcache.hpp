// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_FROMCACHE_HPP
#define POPRITHMS_SCHEDULE_SHIFT_FROMCACHE_HPP

#include <ostream>

#include <poprithms/schedule/shift/ischedulecache.hpp>
#include <poprithms/schedule/shift/kahndecider.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * Obtain a schedule for #graph. The following two methods of obtaining a
 * schedule are attempted, in order:
 *
 * [HOT] Find an exact match of #graph in #rCache, and return the cached
 *       schedule. Only matches which have a RotationTermination condition
 *       equal to the one in #settings are considered.
 *
 * [COLD] Schedule the Graph from scratch.
 *
 * \param graph The Graph to schedule.
 *
 * \param settings The settings to use to schedule the Graph in the case that
 *                 there is no cache hit (COLD).
 *
 * \param writer (optional) A summary of the algorithm's execution and the
 *               graph that it schedules can optionally be stored/written by
 *               #writer. The default is to try and set it from environment
 *               variables if they exist, or else to never write. See
 *               ISummaryWriter for more information.
 *
 * \param rCache The cache to read from, if it is not nullptr. If rCache is
 *               nullptr, then option COLD (see above) is taken.
 *
 * \param wCache The cache to write to, if not nullptr. If wCache is nullptr,
 *               then no solutions are written. Similary, if the HOT path (see
 *               above) is used, then no solution is written.
 *
 * */

ScheduledGraph fromCache(Graph &&graph,
                         const Settings &settings,
                         const ISummaryWriter &writer = FileWriter::Default(),
                         const IScheduleCache *rCache = nullptr,
                         IScheduleCache *wCache       = nullptr);

enum class CacheHitType { Hot, Cold };
std::ostream &operator<<(std::ostream &, CacheHitType);

/**
 * Returns either:
 *  {Cold, {}} if there is cache git for #graph, or
 *  {Hot, schedule} if there is a cache hit with solution is #schedule.
 * */
std::tuple<CacheHitType, std::vector<OpAddress>>
probeCache(const Graph &graph,
           const RotationTermination &,
           const IScheduleCache *cache);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
