// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <array>

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/kahntiebreaker.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

namespace {

std::array<std::string, NKahnTieBreakers> initKahnTieBreakers() {

  constexpr const char *const unset{"unset"};
  std::array<std::string, NKahnTieBreakers> ktbs;
  for (uint64_t i = 0; i < NKahnTieBreakers; ++i) {
    ktbs[i] = unset;
  }
  ktbs[static_cast<uint64_t>(KahnTieBreaker::RANDOM)] = "Random";
  ktbs[static_cast<uint64_t>(KahnTieBreaker::GREEDY)] = "Greedy";
  ktbs[static_cast<uint64_t>(KahnTieBreaker::FIFO)]   = "Fifo";
  for (uint64_t i = 0; i < NKahnTieBreakers; ++i) {
    if (ktbs[i] == unset) {
      throw error("Not all KahnTieBreaker strings are set");
    }
  }
  return ktbs;
}

const std::array<std::string, NKahnTieBreakers> &getKahnTieBreakers() {
  const auto static x = initKahnTieBreakers();
  return x;
}

} // namespace

std::ostream &operator<<(std::ostream &ost, KahnTieBreaker ktb) {
  ost << getKahnTieBreakers()[static_cast<uint64_t>(ktb)];
  return ost;
}

KahnTieBreaker kahnTieBreaker(const std::string &mixedCase) {
  auto lower = util::lowercase(mixedCase);
  for (uint64_t i = 0; i < NKahnTieBreakers; ++i) {
    if (lower == util::lowercase(getKahnTieBreakers()[i])) {
      return static_cast<KahnTieBreaker>(i);
    }
  }
  throw error("Invalid kahnTieBreaker string, " + mixedCase);
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
