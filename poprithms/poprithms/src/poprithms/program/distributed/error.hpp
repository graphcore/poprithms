// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_DISTRIBUTED_HPP
#define POPRITHMS_PROGRAM_DISTRIBUTED_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace program {
namespace distributed {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace distributed
} // namespace program
} // namespace poprithms

#endif
