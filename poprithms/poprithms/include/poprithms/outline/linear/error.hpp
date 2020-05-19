// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_OUTLINE_LINEAR_ERROR_HPP
#define POPRITHMS_OUTLINE_LINEAR_ERROR_HPP

#include <poprithms/util/error.hpp>

namespace poprithms {
namespace outline {
namespace linear {

poprithms::util::error error(const std::string &what);

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif
