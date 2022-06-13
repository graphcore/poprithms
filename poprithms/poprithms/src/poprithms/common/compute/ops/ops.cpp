// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/nop.hpp>

// This file is a collection of implementations of ops which do not have much
// code to implement. This prevents multiple very small translation units
// which can adversely effect compile time.
namespace poprithms {
namespace common {
namespace compute {

/**
 *
 * Nop
 * */
UpOp Nop::cloneWithState(const State &s) const {
  return std::make_unique<Nop>(s);
}

} // namespace compute
} // namespace common
} // namespace poprithms
