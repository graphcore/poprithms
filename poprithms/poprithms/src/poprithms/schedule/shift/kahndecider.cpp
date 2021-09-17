// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <array>

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/kahndecider.hpp>
#include <poprithms/util/printiter.hpp>
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

std::vector<double> KahnDecider::getSparsePriorities(uint64_t nOps) const {
  std::vector<double> allPris(nOps, 0.0);
  for (const auto &p : priorities_) {
    const uint64_t i = std::get<0>(p);
    if (i >= nOps) {
      std::ostringstream oss;
      oss << "Invalid priority index '" << i << "' with nOps = " << nOps
          << '.';
      throw error(oss.str());
    }
    allPris[i] = std::get<1>(p);
  }
  return allPris;
}

std::ostream &operator<<(std::ostream &ost,
                         const std::tuple<OpAddress, double> &p) {
  ost << std::get<0>(p) << ':' << std::get<1>(p);
  return ost;
}

std::ostream &
operator<<(std::ostream &os,
           const std::vector<std::tuple<OpAddress, double>> &t) {
  std::vector<std::string> frags;
  frags.reserve(t.size());
  for (auto x : t) {
    std::ostringstream oss;
    oss << x;
    frags.push_back(oss.str());
  }
  poprithms::util::append(os, frags);
  return os;
}

std::ostream &operator<<(std::ostream &ost, const KahnDecider &kade) {
  kade.append(ost);
  return ost;
}

void KahnDecider::append(std::ostream &ost) const {
  ost << kahnTieBreaker();
  if (!priorities_.empty()) {
    ost << " : ";
    ost << priorities_;
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
