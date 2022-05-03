// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_ERROR_HPP
#define POPRITHMS_COMMON_COMPUTE_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace compute {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
