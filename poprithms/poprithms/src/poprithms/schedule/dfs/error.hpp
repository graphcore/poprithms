// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_DFS_ERROR_HPP
#define POPRITHMS_SCHEDULE_DFS_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace schedule {
namespace dfs {

poprithms::error::error error(const std::string &what);

} // namespace dfs
} // namespace schedule
} // namespace poprithms

#endif
