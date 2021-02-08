// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_BIFURCATE_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_BIFURCATE_HPP

#include <vector>

#include <poprithms/schedule/anneal/graph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

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
// For D = 4;

/*

              o

      o0              o1
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
// All Ops are non-inplace and produce 1 allocation of weight 1.
//
// It is easy to see that the maximum liveness of any schedule is an
// integer in the range [D+2, 2^D+1].
//
// We test that these extrema are obtained with the annealing algorithm.
//

Graph getBifurcatingGraph0(uint64_t D);

// Final max liveness should be D+2
void assertGlobalMinimumBifurcatingGraph0(const Graph &g, uint64_t D);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
