// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COLORING_ERROR_HPP
#define POPRITHMS_COLORING_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace coloring {

poprithms::error::error error(const std::string &what);
poprithms::error::error error(error::Code code, const std::string &what);

} // namespace coloring
} // namespace poprithms

#endif
