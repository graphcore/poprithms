// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_ERROR_HPP
#define POPRITHMS_MEMORY_INPLACE_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif