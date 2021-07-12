// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_MAP_HPP
#define POPRITHMS_UTIL_MAP_HPP

#include <sstream>
#include <vector>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace util {

/**
 * \return a vector of Values. The vector is of the same size as the
 *         iterable container, #keys, and contains values from the Map #m.
 *         Specifically, if #values is the returned vector, then
 *         values[i] = m[keys[i]].
 * */
template <typename Keys, typename Value, typename Map>
std::vector<Value> getValues(const Keys &keys, Map &&m) {
  std::vector<Value> vals;
  vals.reserve(keys.size());
  for (const auto &k : keys) {
    const auto found = m.find(k);
    if (found == m.cend()) {
      std::ostringstream oss;
      oss << "Failed in getValues with Keys=" << keys << ". Did not find key "
          << k << " in the Map 'm' of size " << m.size();
      throw poprithms::error::error("util", oss.str());
    }
    vals.push_back(found->second);
  }
  return vals;
}

} // namespace util
} // namespace poprithms

#endif
