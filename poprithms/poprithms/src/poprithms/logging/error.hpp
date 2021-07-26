// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_LOGGING_ERROR_HPP
#define POPRITHMS_LOGGING_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace logging {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(uint64_t id, const std::string &what);

} // namespace logging
} // namespace poprithms

#endif