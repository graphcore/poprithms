// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_GRAPHSERIALIZATION
#define POPRITHMS_SCHEDULE_SHIFT_GRAPHSERIALIZATION

#include <string>

#include <poprithms/schedule/shift/graph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {
namespace serialization {

Graph fromSerializationString(const std::string &serialization);

} // namespace serialization
} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
