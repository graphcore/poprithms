// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_DIAMOND_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_DIAMOND_HPP

#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

Graph getDiamondGraph0(uint64_t N);

void assertGlobalMinimumDiamondGraph0(const ScheduledGraph &, uint64_t N);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
