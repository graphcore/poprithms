// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_SCHEDULABLE_ERROR_HPP
#define POPRITHMS_COMMON_SCHEDULABLE_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

poprithms::error::error error(const std::string &what);

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
