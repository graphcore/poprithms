// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_OUTLINE_LINEAR_ERROR_HPP
#define POPRITHMS_OUTLINE_LINEAR_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace outline {
namespace linear {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif