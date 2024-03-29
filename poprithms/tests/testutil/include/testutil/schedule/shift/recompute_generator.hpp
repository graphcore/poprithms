// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_RECOMPUTE_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_RECOMPUTE_HPP

#include <vector>

#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

// recompute graphs
//
// example of log-mem graph.
// N = 11
//
//  finish
//    b - b - b < b - b - b - b - b - b - b - b
//    ^   |   |   |   |   |   ^   |   |   |   ^
//    |   |   |   |   |   |   |   |   |   |   |
//    |   x   | / x   |   |   x   | / x   |   |
//    | / x - x - x - x   | / x - x - x - x   |
//    x > x - x - x - x - x - x - x - x - x > x
//  start
//
// n-times computed in forwards section :
//    1   3   2   3   2   1   3   2   3   2   1
//
// see recomp_illustration for a matplotlib generated pdf of the above "plot"
//

std::vector<int> getLogNSeries(uint64_t N);

//
// example of sqrt-mem graph.
// N = 9
//
//
// finish
//   b - b - b < b - b < b - b - b < b
//   ^   |   |   |   |   |   ^   |   |
//   |   |   |   |   |   |   |   |   |
//   |   x   x   x   |   x   x   x   |
//   x > x - x - x - x - x - x - x - x
// start
//
// n-times computed in forwards section :
//   1   2   2   2   1   2   2   2   1
//

std::vector<int> getSqrtSeries(uint64_t N);

// Note: Returned graph has allocations in the range [allocLower, allocUpper)
// : each op creates a random allocation, which is required by all of its
// consumers.
Graph getRecomputeGraph(const std::vector<int> &nTimes,
                        uint64_t allocLower = 1,
                        uint64_t allocUpper = 2,
                        uint32_t seed       = 1011);

// Note: Given graph must have no internal ops. This method assumes allocs are
// all of size 1.
void assertGlobalMinimumRecomputeGraph0(const ScheduledGraph &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
