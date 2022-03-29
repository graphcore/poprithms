// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_VALUEDTUPLE_HPP
#define POPRITHMS_UTIL_VALUEDTUPLE_HPP

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

/**
 * Design pattern: Wrap a tuple, and inherit from the wrapping. This means
 * comparison methods don't need multiple re-implementations.
 * */
template <typename Tup> struct ValuedTuple {
public:
  ValuedTuple(const Tup &tup) : tup_(tup) {}

  template <uint64_t i, typename T> const T &get() const {
    return std::get<i>(tup_);
  }

  template <uint64_t i, typename T> void setVal(T t) {
    std::get<i>(tup_) = t;
  }

  bool operator==(const ValuedTuple &r) const { return tup() == r.tup(); }
  bool operator<(const ValuedTuple &r) const { return tup() < r.tup(); }
  bool operator>(const ValuedTuple &r) const { return tup() > r.tup(); }
  bool operator!=(const ValuedTuple &r) const { return !operator==(r); }
  bool operator<=(const ValuedTuple &rhs) const { return !operator>(rhs); }
  bool operator>=(const ValuedTuple &rhs) const { return !operator<(rhs); }

  const Tup &tup() const { return tup_; }

private:
  Tup tup_;
};

} // namespace util
} // namespace poprithms

#endif
