// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_INTERVAL_HPP
#define POPRITHMS_UTIL_INTERVAL_HPP

#include <array>
#include <vector>

namespace poprithms {
namespace util {

class Interval {
public:
  /**
   * Interval of non-negative integers in [l, u).
   * */
  Interval(uint64_t l, uint64_t u);

  /**
   * Interval lower bound.
   * */
  uint64_t l() const { return std::get<0>(t); }

  /**
   * Interval upper bound.
   * */
  uint64_t u() const { return std::get<1>(t); }

  /**
   * Number of integers in the interval.
   * */
  uint64_t size() const { return u() - l(); }
  int64_t size_i64() const { return static_cast<int64_t>(size()); }

  bool operator==(const Interval &rhs) const { return tup() == rhs.tup(); }
  bool operator!=(const Interval &rhs) const { return !operator==(rhs); }

  /**
   * Lexicographic comparison. Implementing manually, due to C++20 deprecation
   * https://en.cppreference.com/w/cpp/container/array/operator_cmp
   * */
  bool operator<(const Interval &rhs) const {
    return l() < rhs.l() || (l() == rhs.l() && u() < rhs.u());
  }
  bool operator>(const Interval &rhs) const {
    return !operator==(rhs) && !operator>(rhs);
  }
  bool operator<=(const Interval &rhs) const {
    return operator<(rhs) || operator==(rhs);
  }
  bool operator>=(const Interval &rhs) const {
    return operator>(rhs) || operator==(rhs);
  }

  std::array<uint64_t, 2> tup() const { return t; }

  void append(std::ostream &) const;

private:
  std::array<uint64_t, 2> t;
};

/**
 * A set of disjoint intervals.
 * */
class Intervals {
public:
  /**
   * Construct an Intervals object from a set of, not necessarily sorted or
   * disjoint, Intervals. The internally stored Intervals are sorted and
   * disjoint.
   * */
  Intervals(const std::vector<Interval> &);
  static Intervals fromArrays(const std::vector<std::array<uint64_t, 2>> &);

  /**
   * Construct a singleton Intervals.
   * */
  Intervals(uint64_t x0, uint64_t x1);

  bool operator==(const Intervals &rhs) const { return is_ == rhs.is_; }
  bool operator!=(const Intervals &rhs) const { return !operator==(rhs); }
  bool operator<(const Intervals &rhs) const { return is_ < rhs.is_; }
  bool operator>(const Intervals &rhs) const { return is_ > rhs.is_; }
  bool operator<=(const Intervals &rhs) const { return is_ <= rhs.is_; }
  bool operator>=(const Intervals &rhs) const { return is_ >= rhs.is_; }

  /**
   * The total number of elements in the disjoint Intervals.
   * */
  uint64_t size() const { return size_; }
  int64_t size_i64() const { return static_cast<int64_t>(size()); }

  /**
   * A subset of elements based on ranks. Specifically, the subset of
   * elements, from the rank0'th largest to the rank1'th largest, are
   * returned. The rank0'th is included, the rank1'th is excluded, so the
   * number of elements in the Intervals returned is rank1 - rank0.
   *
   * For example, if this is {[2,4), [6,9)}, then subIntervals(1,4) is
   * {[3,4), [6,8)}, as explained below:
   *
   * 0123456789    positive number line
   *   xx  xxx     elements in Intervals [2,4) and [6,9).
   *   01  234     rank of elements in Intervals
   *    |  ||      elements with rank 1,2, and 3: [3,4), [6,8).
   * */
  Intervals subIntervals(int64_t rank0, int64_t rank1) const;

  /**
   * \return true if this interval is [0, a) for some a.
   * */
  bool contiguousFromZero() const;

  const std::vector<Interval> &intervals() const { return is_; }
  Interval interval(uint64_t i) const { return is_[i]; }

  void append(std::ostream &) const;
  std::string str() const;

private:
  std::vector<Interval> is_;
  uint64_t size_;
};

std::ostream &operator<<(std::ostream &, const Interval &);
std::ostream &operator<<(std::ostream &, const Intervals &);

} // namespace util
} // namespace poprithms

#endif
