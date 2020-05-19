// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_DIAMOND_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_DIAMOND_HPP

#include <poprithms/schedule/anneal/graph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

Graph getDiamondGraph0(uint64_t N);

void assertGlobalMinimumDiamondGraph0(const Graph &, uint64_t N);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
