// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <schedule/shift/greedykahn.hpp>
#include <schedule/vanilla/greedystack.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

// We hide the implementation in a translation unit to reduce the size of the
// scheduledgraph.cpp translation unit, which is by the largest in this
// project. This means re-compile times are a bit quicker (this is quite a
// large template to compile).
std::vector<OpAddress>
greedyKahn(const vanilla::Edges<OpAddress> &fwdEdges,
           const vanilla::Priorities<OpAddress, double> &priorities,
           const vanilla::Links<OpAddress> &links,
           const std::vector<AllocWeight> &sizes,
           const vanilla::Edges<OpAddress> &allocsToNodes,
           vanilla::ErrorIfCycle eic,
           vanilla::VerifyEdges ve) {
  return poprithms::schedule::vanilla::greedy::
      kahn<uint64_t, double, AllocWeight>(
          fwdEdges, priorities, links, sizes, allocsToNodes, eic, ve);
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
