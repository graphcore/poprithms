// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_UPDATEFROMFIRSTFINAL_HPP
#define POPRITHMS_SCHEDULE_SHIFT_UPDATEFROMFIRSTFINAL_HPP

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

void updateFromFirstFinal(AllocWeight &lwr,
                          AllocWeight &upp,
                          const AllocWeight &w,
                          const std::tuple<transitiveclosure::IsFirst,
                                           transitiveclosure::IsFinal> ff);

}
} // namespace schedule
} // namespace poprithms

#endif
