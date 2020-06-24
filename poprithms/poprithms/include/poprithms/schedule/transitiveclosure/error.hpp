// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_PATHMATRIX_ERROR_HPP
#define POPRITHMS_SCHEDULE_PATHMATRIX_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

poprithms::error::error error(const std::string &what);

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms

#endif
