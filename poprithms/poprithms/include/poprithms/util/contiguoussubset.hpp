// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_CONTIGUOUSSUBSET_HPP
#define POPRITHMS_UTIL_CONTIGUOUSSUBSET_HPP

#include <cstdint>
#include <sstream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace util {

/**
 * Given a vector of integral values (of type T) 0...#N-1, remove the values
 * in #toRemove, and store them at contiguous indices from 0.
 *
 * Example: N = 4, toRemove = {0,2}.
 *
 * 0...N-1               : 0 1 2 3
 * 0 and 2 removed       :   |   |
 *                           v   v
 *                           1   3
 *                         -------
 * the remaining values  : 1 3.
 * (1,3) are moved to positions (0,1). So the following calls can be expected:
 *
 * inSubset(1) = 0
 * inSubset(3) = 1
 * inFullset(0) = 1
 * inFullset(1) = 3
 * */
template <typename T> class ContiguousSubset {

public:
  static uint64_t get_u64(T t) {
    return poprithms::util::IntValGetter<T>::get_u64(t);
  }

  ContiguousSubset(uint64_t N, const std::vector<T> &toRemove) {

    // a mapping from full set values, to subset values. initialize all subset
    // values as invalid, only full set elements which are not removed will
    // get a value.
    toSubset.resize(N, T(-1));

    isRetainedMask.resize(N, true);
    for (auto i : toRemove) {
      isRetainedMask.at(get_u64(i)) = false;
    }

    uint64_t nxt{0};
    for (uint64_t i = 0; i < N; ++i) {
      if (isRetainedMask[i]) {
        toSubset[i] = nxt;
        toFullset.push_back(i);
        ++nxt;
      }
    }
  }

  // Select the values at retained indices from #us.
  template <class U> void reduce(std::vector<U> &us) const {
    if (us.size() != isRetainedMask.size()) {
      std::ostringstream oss;
      oss << "Incorrect number (" << us.size()
          << ") of elements in reduce from " << isRetainedMask.size()
          << "-element vector.";
      throw poprithms::error::error("util", oss.str());
    }
    for (uint64_t i = 0; i < nSubset(); ++i) {
      auto old = us[get_u64(toFullset[i])];
      us[i]    = std::move(old);
    }
    us.erase(us.cbegin() + nSubset(), us.cend());
  }

  // Explanation by example..
  //
  // Suppose this ContiguousSubset removes at:
  //
  // 0 1 2 3 4 5 6 7 8 9
  // . x x x x . . . . .  (where x == removed).
  //
  // Suppose us = {a,b,c,d,e} and indices = {0,1,2,4,6}:
  //
  // a b c . d . e . . . (the values to filter).
  //   x x x x
  //
  // b, c and d are all at removal indices, so {a,e} is returned.
  template <class U>
  void reduce(std::vector<U> &us, const std::vector<T> &indices) const {
    if (us.size() != indices.size()) {
      throw poprithms::error::error(
          "util", "values and indices vectors are different lengths");
    }
    for (auto v : indices) {
      if (get_u64(v) >= isRetainedMask.size()) {
        throw poprithms::error::error("util", "Invalid index");
      }
    }
    uint64_t currentInsertIndex{0};
    for (uint64_t i = 0; i < us.size(); ++i) {
      if (isRetainedMask[get_u64(indices[i])]) {
        us[currentInsertIndex] = us.at(i);
        ++currentInsertIndex;
      } else {
      }
    }
    us.erase(us.cbegin() + currentInsertIndex, us.cend());
  }

  // The number of elements in the remaining subset
  uint64_t nSubset() const { return toFullset.size(); }

  uint64_t nRemoved() const { return isRetainedMask.size() - nSubset(); }

  T inSubset(T t) const { return toSubset.at(get_u64(t)); }

  T inFullset(T t) const { return toFullset.at(get_u64(t)); }

  bool isRemoved(T t) const { return !isRetainedMask.at(get_u64(t)); }

  std::vector<T> toRemove() const {
    std::vector<T> notRetained;
    notRetained.reserve(isRetainedMask.size() - toFullset.size());
    for (uint64_t i = 0; i < isRetainedMask.size(); ++i) {
      if (!isRetainedMask[i]) {
        notRetained.push_back(i);
      }
    }
    return notRetained;
  }

private:
  std::vector<T> toFullset;
  std::vector<T> toSubset;
  std::vector<bool> isRetainedMask;
};

} // namespace util
} // namespace poprithms

#endif
