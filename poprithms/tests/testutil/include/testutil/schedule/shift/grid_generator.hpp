// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_GRID_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_GRID_HPP

#include <vector>

#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

//                                         (N-1, N-1)
//      o  ->  o  ->  o  ->  z  ->  o  ->  o
//      ^                    =             |
//      |                                 \ /
//      o  ->  o  ->  o  ->  z  ->  o  ->  o ====== most expensive point:
//      ^                    =             |        3 expensives are live
//      |                                 \ /       N-2 cheaps are live.
//      o  ->  o  ->  o  ->  z  ->  o  ->  o
//      ^                    =             |
//      |                                 \ /
//      o  ->  o  ->  o  ->  z  ->  o  ->  o
//      ^                    =             |
//      |                                 \ /
//      o  ->  o  ->  o  ->  z  ->  o  ->  o
//      ^                    =             |
//      |                                 \ /
//      o  ->  o  ->  o  ->  z  ->  o  ->  o
//  (0,0)                    =
//
//  An N x N grid of ops resembling forwards-backwards of an nn.
//  @z alloc is of size 1 @o alloc is of size 2*N
//
//  max should be in [3*2*N + (N-2)*1, O(N^2)]

Graph getGridGraph0(uint64_t rowSize,
                    uint64_t allocLower = 1,
                    uint64_t allocUpper = 2,
                    uint32_t seed       = 1011);

void assertGlobalMinimumGridGraph0(const ScheduledGraph &, uint64_t rowSize);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
