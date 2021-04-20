// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_KAHNTIEBREAKER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_KAHNTIEBREAKER_HPP

#include <cstdlib>
#include <ostream>
#include <string>

namespace poprithms {
namespace schedule {
namespace shift {

/// The core sum-liveness minimizing algorithm is preceded by a single run of
/// Kahn's algorithm to obtain an initial, valid schedule. Kahn's algorithm
/// requires a "tie-breaker" when more than 1 Op is schedulable. Three
/// tie-breakers are implemented:
enum class KahnTieBreaker {
  RANDOM = 0, ///< Choose an Op at random
  GREEDY, ///< Choose the Op which results in the largest immediate liveness
          ///< reduction.
  FIFO,   ///< Choose the Op which became available most recently (this should
          ///< be called FILO).
  N ///< Not a tie-breaker: this is the number of tie-breakers listed above.
};

static constexpr auto NKahnTieBreakers =
    static_cast<uint64_t>(KahnTieBreaker::N);
std::ostream &operator<<(std::ostream &, KahnTieBreaker);
KahnTieBreaker kahnTieBreaker(const std::string &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
