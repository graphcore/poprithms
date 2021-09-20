// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_GREEDYKAHN_HPP
#define POPRITHMS_SCHEDULE_SHIFT_GREEDYKAHN_HPP

#include <array>
#include <tuple>
#include <vector>

#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

std::vector<OpAddress>
greedyKahn(const vanilla::Edges<OpAddress> &fwdEdges,
           const vanilla::Priorities<OpAddress, double> &priorities,
           const vanilla::Links<OpAddress> &links,
           const std::vector<AllocWeight> &sizes,
           const vanilla::Edges<OpAddress> &allocsToNodes,
           vanilla::ErrorIfCycle eic,
           vanilla::VerifyEdges ve);

}
} // namespace schedule
} // namespace poprithms

#endif
