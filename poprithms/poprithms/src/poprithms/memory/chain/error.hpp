// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_ERROR_HPP
#define POPRITHMS_MEMORY_CHAIN_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace memory {
namespace chain {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif