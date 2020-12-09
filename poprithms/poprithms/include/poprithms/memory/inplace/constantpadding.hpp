// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CONSTANT_PADDING_HPP
#define POPRITHMS_MEMORY_INPLACE_CONSTANT_PADDING_HPP
#include <ostream>

namespace poprithms {
namespace memory {
namespace inplace {

/// This enum class defines whether the padding Tensor(s) wrappend around a
/// Tensor are Variable or Constant. see Graph::pad
enum class ConstantPadding {
  No = 0, ///< Pad with Variable Tensor(s)
  Yes     ///< Pad with Constant Tensor(s)
};
std::ostream &operator<<(std::ostream &, ConstantPadding);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
