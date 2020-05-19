// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_ANNEAL_ERROR_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_ERROR_HPP

#include <poprithms/util/error.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::util::error error(const std::string &what);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
