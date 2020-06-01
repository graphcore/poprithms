// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_NEST_ERROR_HPP
#define POPRITHMS_MEMORY_NEST_ERROR_HPP

#include <poprithms/util/error.hpp>

namespace poprithms {
namespace memory {
namespace nest {

poprithms::util::error error(const std::string &what);

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
