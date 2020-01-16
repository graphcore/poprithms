#ifndef POPRITHMS_SCHEDULE_ANNEAL_PRINTITER_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_PRINTITER_HPP

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
