// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_ERROR_HPP
#define POPRITHMS_UTIL_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace util {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(uint64_t id, const std::string &what);

} // namespace util
} // namespace poprithms

#endif