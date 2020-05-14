#include <initializer_list>
#include <string>

#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

template void append<>(std::ostream &, const std::vector<int64_t> &);
template void append<>(std::ostream &, const std::vector<uint64_t> &);
template void append<>(std::ostream &, const std::vector<int> &);
template void append<>(std::ostream &, const std::vector<std::string> &);

} // namespace util
} // namespace poprithms
