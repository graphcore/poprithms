// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_ERROR_HPP
#define POPRITHMS_NDARRAY_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace ndarray {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(uint64_t id, const std::string &what);

} // namespace ndarray
} // namespace poprithms

#endif