// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_LOGGING_ERROR_HPP
#define POPRITHMS_UTIL_LOGGING_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace logging {

poprithms::error::error error(const std::string &what);

} // namespace logging
} // namespace poprithms

#endif
