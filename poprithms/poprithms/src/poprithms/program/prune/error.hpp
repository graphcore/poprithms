// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PRUNE_ERROR_HPP
#define POPRITHMS_PROGRAM_PRUNE_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace program {
namespace prune {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace prune
} // namespace program
} // namespace poprithms

#endif
