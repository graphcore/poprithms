// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <sstream>

#include <poprithms/util/error.hpp>
#include <poprithms/util/interval.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

namespace {

// Example if in0 ={[2,5), [0,4)}, then {[0, 5)} is returned:
//
//      012345
//        ...   [2,5)
//      ....    [0,4)
//      .....   [0,5)
//
std::vector<Interval> sortedUnion(const std::vector<Interval> &in0) {
  auto nxt = in0;
  std::sort(nxt.begin(), nxt.end());
  bool merged{true};
  std::vector<Interval> curr;
  while (merged) {
    std::swap(curr, nxt);
    nxt.clear();
    merged = false;
    for (const auto &i : curr) {

      if (i.size() == 0) {
        continue;
      }

      // if nxt has no Intervals, or if there's space between the end of the
      // last Interval of nxt and i:
      if (nxt.empty() || nxt.back().u() < i.l()) {
        nxt.push_back(i);
      }

      // else, merge i into the back of nxt:
      // Note that the sorting is not undone here, as
      // a <= b <= c implies that
      // union(a,b) <= c.
      else {
        merged     = true;
        nxt.back() = {nxt.back().l(), std::max(nxt.back().u(), i.u())};
      }
    }
  }
  return nxt;
}

std::vector<Interval>
intervalsFromArrays(const std::vector<std::array<uint64_t, 2>> &arrs) {
  std::vector<Interval> is;
  is.reserve(arrs.size());
  for (auto a : arrs) {
    is.push_back({std::get<0>(a), std::get<1>(a)});
  }
  return is;
}

} // namespace

bool Intervals::contiguousFromZero() const {
  return (*this == Intervals(0, size()));
}

Intervals Intervals::subIntervals(int64_t r0, int64_t r1) const {

  std::vector<Interval> out;

  // How many elements have been seen so far, moving from the lowest
  // (left-most) interval to the highest (right-most).
  int64_t currentRank{0};

  for (auto I : is_) {

    // The length of the current interval
    const int64_t L        = I.size_i64();
    const int64_t nextRank = currentRank + L;
    const auto start       = r0 > currentRank ? r0 - currentRank : 0;
    const auto end         = r1 > nextRank ? L : (L - (nextRank - r1));

    if (start < L && end > start) {
      out.push_back({I.l() + start, I.l() + end});
    }

    currentRank = nextRank;
  }

  return Intervals(out);
}

Interval::Interval(uint64_t l, uint64_t u) : t({l, u}) {
  if (u < l) {
    std::ostringstream oss;
    oss << "Invalid interval, l = " << l << ", u = " << u
        << ". Expected u >= l.";
    throw error(oss.str());
  }
}

std::string Intervals::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

Intervals
Intervals::fromArrays(const std::vector<std::array<uint64_t, 2>> &ses) {
  return Intervals(intervalsFromArrays(ses));
}

Intervals::Intervals(const std::vector<Interval> &ses)
    : is_(sortedUnion(ses)) {
  size_ = 0;
  for (const auto &se : is_) {
    size_ += se.size();
  }
}

Intervals::Intervals(uint64_t x0, uint64_t x1)
    : Intervals(std::vector<Interval>{{x0, x1}}) {}

void Interval::append(std::ostream &ost) const {
  ost << '[' << l() << ',' << u() << ')';
}

void Intervals::append(std::ostream &ost) const { util::append(ost, is_); }

std::ostream &operator<<(std::ostream &ost, const Interval &i) {
  i.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Intervals &is) {
  is.append(ost);
  return ost;
}

} // namespace util
} // namespace poprithms
