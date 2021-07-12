// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SCC_ERROR_HPP
#define POPRITHMS_SCHEDULE_SCC_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace schedule {
namespace scc {

poprithms::error::error error(const std::string &what);

} // namespace scc
} // namespace schedule
} // namespace poprithms

#endif
