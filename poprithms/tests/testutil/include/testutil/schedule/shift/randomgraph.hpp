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
//
// comes with Allocs applied: Each Op creates 1 Alloc in the range [lowAlloc,
// highAlloc), which is used by all of its consumers.
Graph getRandomGraph(uint64_t N,
                     uint64_t E,
                     uint64_t D,
                     int32_t graphSeed,
                     uint64_t lowAlloc  = 10,
                     uint64_t highAlloc = 20);

// Every node in the graph has an allocation which it is the 'creator' of,
// with a size in the range [lowAlloc, highAlloc). All consuers of the op have
// the alloc assigned to them too.
void addConnectedAllocs(Graph &g,
                        uint64_t lowAlloc,
                        uint64_t highAlloc,
                        uint32_t seed);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
