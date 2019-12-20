#include <poprithms/schedule/anneal/printiter.hpp>

namespace poprithms {
namespace util {
namespace {

template <typename T> void tAppend(std::ostream &os, const T &t) {
  os << '(';
  if (t.size() > 0) {
    auto t0 = std::cbegin(t);
    os << *t0;
    ++t0;
    while (t0 != std::cend(t)) {
      os << ',' << *t0;
      ++t0;
    }
  }
  os << ')';
}

} // namespace

void append(std::ostream &os, const std::vector<uint64_t> &v) {
  tAppend(os, v);
}
void append(std::ostream &os, const std::vector<int64_t> &v) {
  tAppend(os, v);
}
void append(std::ostream &os, const std::vector<int> &v) { tAppend(os, v); }
} // namespace util
} // namespace poprithms
