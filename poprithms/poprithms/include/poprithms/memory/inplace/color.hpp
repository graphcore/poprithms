// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_COLOR_HPP
#define POPRITHMS_MEMORY_INPLACE_COLOR_HPP

#include <poprithms/memory/alias/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

static const alias::Color ConstantColor = 0;
static const alias::Color VariableColor = 1;

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
