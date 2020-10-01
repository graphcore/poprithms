// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_BOOLIMPL_HPP
#define POPRITHMS_COMPUTE_HOST_BOOLIMPL_HPP

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>

#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * This class is used to circumvent issues with the std::vector<bool> class,
 * arising from its bitwise representation of values:
 *
 * 1) std::vector<bool> does not have a .data() method
 * 2) parallelization is tricky, forgetting to use atomics leads to bugs.
 * */
struct BoolImpl {
  BoolImpl() = default;
  BoolImpl(bool x) : v(x) {}
  template <typename T> operator T() const { return static_cast<T>(v); }
  bool v;
};

inline bool operator>(BoolImpl a, BoolImpl b) { return a.v > b.v; }
inline bool operator>=(BoolImpl a, BoolImpl b) { return a.v >= b.v; }
inline bool operator==(BoolImpl a, BoolImpl b) { return a.v == b.v; }
inline bool operator!=(BoolImpl a, BoolImpl b) { return a.v != b.v; }
inline bool operator<=(BoolImpl a, BoolImpl b) { return a.v <= b.v; }
inline bool operator<(BoolImpl a, BoolImpl b) { return a.v < b.v; }
std::ostream &operator<<(std::ostream &, BoolImpl);

inline BoolImpl operator*(BoolImpl a, BoolImpl b) {
  return BoolImpl(a.v && b.v);
}

inline BoolImpl operator&&(BoolImpl a, BoolImpl b) {
  return BoolImpl(a.v && b.v);
}

inline BoolImpl operator+(BoolImpl a, BoolImpl b) {
  return BoolImpl(a.v || b.v);
}

inline BoolImpl operator||(BoolImpl a, BoolImpl b) {
  return BoolImpl(a.v || b.v);
}

} // namespace host
} // namespace compute

namespace ndarray {
template <> DType get<compute::host::BoolImpl>();
}
} // namespace poprithms

#endif
