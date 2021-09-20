// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_BIFURCATE_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_BIFURCATE_HPP

#include <vector>

#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

// A graph of Ops, where at each depth there are
// d = 0   : 1 Op with 0 producers and 2 consumers
// d = 1   : 2 Ops with 1 producer and 2 consumers
// d = 2   : 4 Ops with 1 producer and 2 consumers
// .
// .
// d = D   : 2^D Ops with 1 producer and 1 consumer
// d = D+1 : 2^(D-1) Ops with 2 producers and 1 consumer
// d = D+2 : 2^(D=2) Ops with 2 producers and 1 consumer
// .
// .
// d = 2D-1 : 1 Op with 2 producers and 0 consumers
//
// For D = 4:

/*
               o
               +-->-+
                     \
      o0              o1
                    /    \
  o00     o01    o10      o11
                          / \
o   o   o   o   o   o    o   o
                            / \
o o o o o o o o o o o o o o o   o
                            \ /
o   o   o   o   o   o    o   o
                          \ /
  o       o       o        o

      o               o
              o

*/

//
// All Ops are non-inplace and produce 1 allocation of weight in range
// [allocLower, allocUpper), seleceted randomly based on a random seed.
//
// It is easy to see that the maximum liveness of any schedule is an
// integer in the range [D+2, 2^D+1].
//
// We test that these extrema are obtained with the shifting algorithm.
//

Graph getBifurcatingGraph0(uint64_t D,
                           uint64_t allocLower = 1,
                           uint64_t allocUpper = 2,
                           uint32_t seed       = 1011);

// Final max liveness should be D+2. For this, the graph should have all
// allocations of size 1.
void assertGlobalMinimumBifurcatingGraph0(const ScheduledGraph &g,
                                          uint64_t D);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
