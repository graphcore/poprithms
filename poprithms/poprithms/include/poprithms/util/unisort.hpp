#ifndef POPRITHMS_UTIL_UNISORT_HPP
#define POPRITHMS_UTIL_UNISORT_HPP

#include <algorithm>
#include <vector>

namespace poprithms {
namespace util {

template <typename T> std::vector<T> unisorted(const std::vector<T> &x) {
  std::vector<T> y = x;
  std::sort(y.begin(), y.end());
  auto last = std::unique(y.begin(), y.end());
  y.erase(last, y.cend());
  return y;
}

} // namespace util
} // namespace poprithms

#endif
