#ifndef TESTUTIL_SCHEDULE_PATHMATRIX_RANDOMGRAPH_HPP
#define TESTUTIL_SCHEDULE_PATHMATRIX_RANDOMGRAPH_HPP

#include <vector>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

std::vector<std::vector<uint64_t>>
getRandomEdges(uint64_t N, uint64_t E, uint64_t D, int graphSeed);

} // namespace pathmatrix
} // namespace schedule
} // namespace poprithms

#endif
