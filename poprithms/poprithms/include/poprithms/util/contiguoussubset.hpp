// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_CONTIGUOUSSUBSET_HPP
#define POPRITHMS_UTIL_CONTIGUOUSSUBSET_HPP

#include <cstdint>
#include <vector>

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
    for (uint64_t i = 0; i < nSubset(); ++i) {
      auto old = us[get_u64(toFullset[i])];
      us[i]    = std::move(old);
    }
    us.erase(us.cbegin() + nSubset(), us.cend());
  }

  uint64_t nSubset() const { return toFullset.size(); }

  T inSubset(T t) const { return toSubset.at(get_u64(t)); }

  T inFullset(T t) const { return toFullset.at(get_u64(t)); }

  bool isRemoved(T t) const { return !isRetainedMask.at(get_u64(t)); }

private:
  std::vector<T> toFullset;
  std::vector<T> toSubset;
  std::vector<bool> isRetainedMask;
};

} // namespace util
} // namespace poprithms

#endif
