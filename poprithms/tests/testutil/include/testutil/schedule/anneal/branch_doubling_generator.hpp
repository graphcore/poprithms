// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef TESTUTIL_SCHEDULE_ANNEAL_BRANCH_DOUBLING_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_BRANCH_DOUBLING_HPP

#include <poprithms/schedule/anneal/graph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

//  Example with nBranches = 3, offset = -1
//
//       root-- Op_2_0 - Op_2_1
//      /   \               |
//  Op_0_0  Op_1_0          .
//    |       \             .
//  Op_0_1   Op_1_1         .
//    |         \           .
//  Op_0_2     Op_1_2    Op_2_6
//     \          |         |
//     End_0 -> End_1 --> End_2
//
// where Op_a_b above is the b'th Op on branch a.
//
// Branch 0 (on left) always has the same number of Ops: 3
// All subsequent branches have nOps() - 1 + offset Ops, so
// branch 1 has 5 - 1 - 1 = 3, and then
// branch 2 has 9 - 1 - 1 = 7 Ops.
//
// Example with nBranches = 6, offset = +1
//
//            root -------
//           /  \   ....   \
//          |    |          |
//      Op_0_0 Op_1_0 ... Op_5_0
//        |      |          |
//        .      .          .
//        .      .          .
//      Op_0_2 Op_1_4 ... Op_5_94
//         |     |          |
//       End_0->End_1.. -> End_5
//
//  Branch lengths are : 3 5 11 23 47 95

Graph getBranchDoublingGraph(uint64_t nBranches, uint64_t offset);

// if offset < 0, expect branches scheduled in ascending order from 0.
// if offset > 0, expect branches scheduled in order
// nBranches - 2, nBranches - 1 ... 0, nBranches - 1.
//
void assertGlobalMinimumBranchDoubling(const Graph &,
                                       int nBranches,
                                       int offset);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
