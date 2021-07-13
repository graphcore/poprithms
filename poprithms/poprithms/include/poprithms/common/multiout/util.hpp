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

/**
 * Given a map M0 with keys as TensorIds and values as some numeric type,
 * accumulate the values by each tensors producing op. For example, if #m0 is
 *  {(0,0):5, (0,1):6, (1,0):3} the returned map is {0:11, 1:3}.
 * */
template <typename M0, typename M1> M1 sumOverOutTensors(const M0 &m0) {
  M1 m1;
  for (const auto &key_value : m0) {
    const auto &k = key_value.first;
    const auto &v = key_value.second;
    auto found    = m1.find(k.opId());
    if (found == m1.cend()) {
      m1.insert({k.opId(), v});
    } else {
      found->second += v;
    }
  }
  return m1;
}

} // namespace util
} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
