// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_ERROR_HPP
#define POPRITHMS_COMPUTE_HOST_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace compute {
namespace host {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace host
} // namespace compute
} // namespace poprithms

#endif