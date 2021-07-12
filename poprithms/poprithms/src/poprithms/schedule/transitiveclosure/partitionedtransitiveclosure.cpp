// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <schedule/transitiveclosure/error.hpp>
#include <sstream>

#include <poprithms/schedule/transitiveclosure/partitionedtransitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

bool PartitionedTransitiveClosure::constrained(OpId from, OpId to) const {

  // If 'from' and 'to' are in different components, there is no constraint
  // between them:
  const auto c = ccs.componentId(from);
  if (c != ccs.componentId(to)) {
    return false;
  }

  // If 'from' and 'to' are in the same component, query the component to see
  // if there is a constraint from->to.
  return transitiveClosures[c.get()].constrained(ccs.localId(from).get(),
                                                 ccs.localId(to).get());
}

PartitionedTransitiveClosure::PartitionedTransitiveClosure(
    const Edges &forwardEdges)
    : ccs(forwardEdges) {
  for (uint64_t c = 0; c < ccs.nComponents(); ++c) {
    transitiveClosures.push_back(TransitiveClosure(ccs.component(c)));
  }
}

uint64_t PartitionedTransitiveClosure::nBits() const {

  // The sum of all the bitset sizes of the the individual transitive
  // closures.
  return std::accumulate(
      transitiveClosures.cbegin(),
      transitiveClosures.cend(),
      0ULL,
      [](uint64_t c, const TransitiveClosure &tc) { return c + tc.nBits(); });
}

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
