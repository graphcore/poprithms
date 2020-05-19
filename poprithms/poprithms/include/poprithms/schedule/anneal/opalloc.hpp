// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_ANNEAL_OPALLOC_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_OPALLOC_HPP

#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

class OpAlloc {
public:
  OpAlloc(OpAddress o, AllocAddress a) : op(o), alloc(a) {}
  OpAddress op;
  AllocAddress alloc;
};

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
