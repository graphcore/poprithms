#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/printiter.hpp>

namespace poprithms {
namespace util {
namespace {

template <typename T, typename V>
void tvAppend(std::ostream &os, const T &t, const V &v) {
  if (t.size() > 0) {
    auto t0 = std::cbegin(t);
    os << v(*t0);
    ++t0;
    while (t0 != std::cend(t)) {
      os << ',' << v(*t0);
      ++t0;
    }
  }
}

template <typename T> class Id {
public:
  T operator()(const T &t) const { return t; }
};

template <typename T> void tAppend(std::ostream &os, const T &t) {
  using E = typename T::value_type;
  tvAppend(os, t, Id<E>());
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
