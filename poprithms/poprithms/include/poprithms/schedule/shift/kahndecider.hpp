// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_KAHNDECIDER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_KAHNDECIDER_HPP

#include <cstdlib>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/schedule/shift/shiftusings.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/// The core sum-liveness minimizing algorithm is preceded by a single run of
/// Kahn's algorithm to obtain an initial, valid schedule. Kahn's algorithm
/// requires a "tie-breaker" decision when more than 1 Op is schedulable.
/// Three tie-breakers are implemented:
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

/**
 * When performing Kahn's algorithm to obtain an initial schedule of a DAG,
 * there are, at times, several Ops which can be moved from a set of
 * 'schedulable' Ops, to the actual, final, schedule. This class determines
 * which of these 'schedulable' Ops is selected.
 *
 * All the Ops have a 'priority'. This is a floating point value which
 * defaults to zero. If an Op in the 'schedulable' set does not have the
 * highest priority, it will not be transferred to the final schedule, until
 * it does. Only the Op(s) which have the highest priority are candidates to
 * be moved into the final schedule.
 *
 * The final decision of which one of the Ops with the highest priority value
 * is scheduled is made based on the KahnkTieBreaker. This means that if all
 * the Ops in the DAG have the same priority value, the decision of which
 * schedulable Op is scheduled is made based purely on the KahnTieBreaker. If
 * all of the Ops have distinct priorities, then the KahnTieBreaker has no
 * effect.
 * */
class KahnDecider {

public:
  using Priority   = std::tuple<OpAddress, double>;
  using Priorities = std::vector<Priority>;

  /**
   * \param priorities A vector of size less than or equal to the number of
   *                   Ops in the DAG to be scheduled. Ops which do not have
   *                   a priority, will receive the default priority value of
   *                   0.0.
   * */
  KahnDecider(KahnTieBreaker ktb, const Priorities &priorities)
      : ktb_(ktb), priorities_(priorities) {}

  void setPriorities(const Priorities &ps) { priorities_ = ps; }

  /**
   * Create a KahnDecider where all the Ops have priority 0.0.*/
  explicit KahnDecider(KahnTieBreaker ktb) : KahnDecider(ktb, {}) {}

  KahnTieBreaker kahnTieBreaker() const { return ktb_; }

  const Priorities &priorities() const { return priorities_; }

  /**
   * \return A vector of size #nOps, where all values are zero except for
   *         those with a priority set for this KahnTieBreaker. #nOps must be
   *         larger than all OpIds in priorities_.
   * */
  std::vector<double> getSparsePriorities(uint64_t nOps) const;

  uint64_t nPrioritized() const { return priorities_.size(); }

  void append(std::ostream &) const;

  bool operator==(const KahnDecider &rhs) const {
    return ktb_ == rhs.ktb_ && (priorities_ == rhs.priorities_);
  }
  bool operator<(const KahnDecider &rhs) const {
    if (ktb_ == rhs.ktb_) {
      return priorities_ < rhs.priorities_;
    }
    return ktb_ < rhs.ktb_;
  }

private:
  KahnTieBreaker ktb_;
  Priorities priorities_;
};

std::ostream &operator<<(std::ostream &ost, const KahnDecider &);
std::ostream &operator<<(std::ostream &ost, const KahnDecider::Priority &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
