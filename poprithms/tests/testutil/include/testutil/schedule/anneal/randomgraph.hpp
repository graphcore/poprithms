// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_RANDOMGRAPH_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_RANDOMGRAPH_HPP

#include <poprithms/schedule/anneal/graph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

// Note: Inserts no internal ops.
Graph getRandomGraph(uint64_t N, uint64_t E, uint64_t D, int graphSeed);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
