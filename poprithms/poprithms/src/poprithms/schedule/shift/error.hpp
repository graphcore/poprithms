// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_ERROR_HPP
#define POPRITHMS_SCHEDULE_SHIFT_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

poprithms::error::error error(const std::string &what);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
