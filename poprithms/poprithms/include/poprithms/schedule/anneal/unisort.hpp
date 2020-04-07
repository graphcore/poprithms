#ifndef POPRITHMS_SCHEDULE_ANNEAL_UINSORT_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_UNISORT_HPP

#include <algorithm>
#include <vector>

namespace poprithms {
namespace schedule {
namespace anneal {

template <typename T> std::vector<T> unisorted(const std::vector<T> &x) {
  std::vector<T> y = x;
  std::sort(y.begin(), y.end());
  auto last = std::unique(y.begin(), y.end());
  y.erase(last, y.cend());
  return y;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
