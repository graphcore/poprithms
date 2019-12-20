#ifndef GUARD_ERROR_UTIL_PRINTITER_HPP
#define GUARD_ERROR_UTIL_PRINTITER_HPP

#include <sstream>
#include <vector>

namespace poprithms {
namespace util {

void append(std::ostream &, const std::vector<int64_t> &);
void append(std::ostream &, const std::vector<int> &);
void append(std::ostream &, const std::vector<uint64_t> &);

} // namespace util
} // namespace poprithms

#endif
