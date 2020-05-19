// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_TRANSITIVECLOSURE_RANDOMGRAPH_HPP
#define TESTUTIL_SCHEDULE_TRANSITIVECLOSURE_RANDOMGRAPH_HPP

#include <vector>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

std::vector<std::vector<uint64_t>>
getRandomEdges(uint64_t N, uint64_t E, uint64_t D, int graphSeed);

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms

#endif
