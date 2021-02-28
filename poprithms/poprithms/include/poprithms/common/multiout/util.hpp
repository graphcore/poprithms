// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_UTIL_UTIL_HPP
#define POPRITHMS_COMMON_MULTIOUT_UTIL_UTIL_HPP

#include <vector>

namespace poprithms {
namespace common {
namespace multiout {
namespace util {

template <typename Ts, typename Ids> Ids ids(const Ts &ts) {
  Ids ids_;
  ids_.reserve(ts.size());
  for (const auto &t : ts) {
    ids_.push_back(t.id());
  }
  return ids_;
}

} // namespace util
} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
