// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <memory/nest/error.hpp>

#include <poprithms/memory/nest/stripe.hpp>

namespace poprithms {
namespace memory {
namespace nest {

namespace {
// adjust phase so that 0 <= phase < period.
int64_t getCorrectedPhase(int64_t phase, int64_t period) {
  if (phase < 0) {
    phase -= period * (phase / period - 1);
  }
  return phase % period;
}
} // namespace

void Stripe::append(std::ostream &ost) const {
  ost << '(' << on() << ',' << off() << ',' << phase() << ')';
}

int64_t Stripe::nOn(int64_t start, int64_t end) const {
  if (end < start) {
    std::ostringstream oss;
    oss << "Stripe::nOn(start=" << start << ',' << "end=" << end
        << ") invalid, call requires that end >= start.";
    throw error(oss.str());
  }
  // contributions from "full periods"
  int64_t total = on() * ((end - start) / period());

  // what's left after "full periods" are accounted for
  auto delta = (end - start) % period();

  // set phase to zero
  start = getCorrectedPhase(start - phase(), period());
  end   = start + delta;

  if (end > period()) {
    total += std::min(end - period(), on());
  }
  if (start < on()) {
    total += std::min(end - start, on() - start);
  }
  return total;
}

int64_t Stripe::nFullPeriods(int64_t begin, int64_t end) const {
  auto ph0   = getCorrectedPhase(phase() - begin, period());
  auto end0  = end - begin;
  auto nFull = std::max(int64_t(0), (end0 - ph0) / period());
  return nFull;
}

int64_t Stripe::firstStartNotBefore(int64_t x) const {
  return x + getCorrectedPhase(phase() - x, period());
}

int64_t Stripe::lastStartNotAfter(int64_t x) const {
  auto nb = firstStartNotBefore(x);
  return nb == x ? nb : nb - period();
}

Stripe::Stripe(int64_t on, int64_t off, int64_t phaseIn)
    : sOn(on), sOff(off), sPhase(phaseIn) {
  if (on == 0 && off == 0) {
    std::ostringstream oss;
    oss << "Stripe::Stripe(on = 0, off = 0, phase = " << phase()
        << ") not allowed, must have period (on + off) > 0";
    throw error(oss.str());
  }

  if (on < 0 || off < 0) {
    throw error("only positive on and off values are currently supported");
  }

  // ensure 0 <= sPhase < outer
  sPhase = getCorrectedPhase(sPhase, period());
}

std::ostream &operator<<(std::ostream &ost, const Stripe &s) {
  s.append(ost);
  return ost;
}

} // namespace nest
} // namespace memory
} // namespace poprithms
