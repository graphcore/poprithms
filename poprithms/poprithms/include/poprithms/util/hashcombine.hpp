// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_HASHCOMBINE_HPP
#define POPRITHMS_UTIL_HASHCOMBINE_HPP

#include <functional>

namespace poprithms {
namespace util {

/**
 * Merge the hash of #v into the (running) hash value, #seed.
 * */
template <class T> inline void hash_combine(std::size_t &seed, const T &v) {
  // 2^32 / phi = 0x9e3779b9 for the curious. Chosen as a random sequence of
  // bits by someone, now everyone (eg boost) seems to use it.
  // [from stackoverflow question 4948780]
  std::hash<T> hasher_;
  seed ^= hasher_(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace util
} // namespace poprithms

#endif
