// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_LOGGING_ERROR_HPP
#define POPRITHMS_LOGGING_ERROR_HPP

#include <poprithms/util/error.hpp>

namespace poprithms {
namespace logging {

poprithms::util::error error(const std::string &what);

} // namespace logging
} // namespace poprithms

#endif
