// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace util {

std::string lowercase(const std::string &x) {
  auto lower = x;
  std::transform(lower.begin(),
                 lower.end(),
                 lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower;
}

std::string spaceString(uint64_t target, const std::string &ts) {
  uint64_t taken = ts.size();
  if (taken > target) {
    return std::string(" ");
  }
  return std::string(target - taken + 1, ' ');
}

} // namespace util
} // namespace poprithms
