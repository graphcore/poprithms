// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_ANNEAL_GRAPHSERIALIZATION
#define POPRITHMS_SCHEDULE_ANNEAL_GRAPHSERIALIZATION

#include <poprithms/schedule/anneal/graph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {
namespace serialization {

Graph fromSerializationString(const std::string &serialization);

} // namespace serialization
} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
