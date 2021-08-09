// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_RANDOMGRAPH_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_RANDOMGRAPH_HPP

#include <poprithms/schedule/shift/graph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

// Note: Inserts no internal ops.
// N : number of nodes
// E : number of input edges per node (excluding the first D nodes).
// D : the input edges are all from an Op between D and 1 back.
Graph getRandomGraph(uint64_t N, uint64_t E, uint64_t D, int graphSeed);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
