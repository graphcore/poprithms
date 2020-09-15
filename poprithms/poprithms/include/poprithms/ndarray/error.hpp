// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_ERROR_HPP
#define POPRITHMS_NDARRAY_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace ndarray {

poprithms::error::error error(const std::string &what);

} // namespace ndarray
} // namespace poprithms

#endif
