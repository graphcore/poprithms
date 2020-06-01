// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_NEST_STRIPE_HPP
#define POPRITHMS_MEMORY_NEST_STRIPE_HPP

#include <cstdint>
#include <sstream>

namespace poprithms {
namespace memory {
namespace nest {

/** A periodic map from integers to {0, 1} represented by 3 values: on, off,
 * and phase.
 *
 * Letting '.' denote 0 for improved diagrams, here are some examples.
 *
 * Example 1: {on, off, phase} = {2, 3, 0}
 *  (on for 2, then off 3, repeated, with no offset).
 *  0 1 2 3 4 5 6 7 8 9 etc.
 *  1 1 . . . 1 1 . . . 1 1 . . . 1 1 . . . 1 1 . .
 *
 * Example 2: {on, off, phase} = {4, 2, 1}
 * (on for 4, then off for 2, repeated, with offset of 1).
 * 0 1 2 3 4 5 6 7 8 9 etc.
 * . 1 1 1 1 . . 1 1 1 1 . . 1 1 1 1 . . 1 1 1 1 . . 1 1 1 1 . .
 */
class Stripe {
public:
  /**Construct a Stripe
   *
   * \param on The number of contiguous integers for which the Stripe is '1'.
   *           Must be non-negative.
   * \param off The number of contiguous integers for which the Stripe's '0'.
   *            Must be non-negative.
   * \param phase The offset from 0 to the first '1'.
   */
  Stripe(int64_t on, int64_t off, int64_t phase);

  int64_t on() const { return sOn; }
  int64_t off() const { return sOff; }
  int64_t period() const { return sOn + sOff; }
  int64_t phase() const { return sPhase; }

  /** The number of full on-off segments in the range [start, end). As an
   * example, suppose {on, off, phase} = {2, 3, 1}:
   *
   * 0 1 2 3 4 5 6 7 8 9
   * . 1 1 . . . 1 1 . . . 1 1 . . . 1 1
   *
   * nFullPeriods(1,6) = 1
   * nFullPeriods(1,5) = 0
   * nFullPeriods(2,9) = 0
   * nFullPeriods(0,9) = 1
   */
  int64_t nFullPeriods(int64_t start, int64_t end) const;

  /**  smallest y such that y >= x and y = phase + k*(on + off) */
  int64_t firstStartNotBefore(int64_t x) const;

  /** largest y such that y <= x and y = phase * k*period */
  int64_t lastStartNotAfter(int64_t x) const;

  /** The number of integers in [start, end) where the Stripe is on ('1') */
  int64_t nOn(int64_t start, int64_t end) const;
  bool alwaysOn() const { return off() == 0; }

  /** Returns true iff all Stripe values for integers in [x, y) are on ('1')
   */
  bool allOn(int64_t x, int64_t y) const {
    return y <= lastStartNotAfter(x) + on();
  }

  /** Returns true iff all Stripe values for integers in [x, y) are off ('0')
   */
  bool allOff(int64_t x, int64_t y) const {
    return firstStartNotBefore(x) >= y && (x - lastStartNotAfter(x) >= on());
  }

  bool operator==(const Stripe &rhs) const {
    return on() == rhs.on() && off() == rhs.off() && phase() == rhs.phase();
  }
  bool operator!=(const Stripe &rhs) const { return !operator==(rhs); }
  Stripe getScaled(int64_t f) const {
    return Stripe(sOn * f, sOff * f, sPhase * f);
  }

  void append(std::ostream &) const;

  uint64_t on_u64() const { return static_cast<uint64_t>(on()); }
  uint64_t off_u64() const { return static_cast<uint64_t>(off()); }
  uint64_t period_u64() const { return static_cast<uint64_t>(period()); }

private:
  int64_t sOn;
  int64_t sOff;
  int64_t sPhase;
};

std::ostream &operator<<(std::ostream &stream, const Stripe &);

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
