// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_OPALLOC_HPP
#define POPRITHMS_SCHEDULE_SHIFT_OPALLOC_HPP

#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class OpAlloc {
public:
  OpAlloc(OpAddress o, AllocAddress a) : op(o), alloc(a) {}
  OpAddress op;
  AllocAddress alloc;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
