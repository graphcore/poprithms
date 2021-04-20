// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_ROTATIONALGO_HPP
#define POPRITHMS_SCHEDULE_SHIFT_ROTATIONALGO_HPP

#include <ostream>

namespace poprithms {
namespace schedule {
namespace shift {

/// Implementations of the sum-liveness minimizing algorithm. They differ only
/// in time to solution, the final schedule obtained using these is identical.
/// RIPPLE is much faster.
enum class RotationAlgo {
  SIMPLE, ///< A simple implementation, for debugging and understanding
  RIPPLE  ///< An optimized implementation which eliminates certain redundant
          ///< computations. It's name derives from the way it re-uses results
          ///< using dynamic programming across consecutive schedule indices.
};

std::ostream &operator<<(std::ostream &, RotationAlgo);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
