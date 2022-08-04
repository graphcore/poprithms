// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PIPELINE_ERROR_HPP
#define POPRITHMS_PROGRAM_PIPELINE_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace pipeline
} // namespace program
} // namespace poprithms

#endif
