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

} // namespace util
} // namespace poprithms
