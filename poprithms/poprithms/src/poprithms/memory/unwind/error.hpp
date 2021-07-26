// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_ERROR_HPP
#define POPRITHMS_MEMORY_UNWIND_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(uint64_t id, const std::string &what);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif