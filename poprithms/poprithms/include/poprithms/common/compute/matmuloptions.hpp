// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_MATMULOPTIONS_HPP
#define POPRITHMS_COMMON_COMPUTE_MATMULOPTIONS_HPP

#include <vector>

namespace poprithms {
namespace common {
namespace compute {

struct MatMulOptions {
  // Currently there are no options for matmul, this is a just placeholder for
  // the future.
public:
  bool operator==(const MatMulOptions &) const { return true; }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
