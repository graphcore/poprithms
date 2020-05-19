// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_PRINTITER_HPP
#define POPRITHMS_UTIL_PRINTITER_HPP

#include <sstream>
#include <vector>

namespace poprithms {
namespace util {

// If t = {1,2,4,5}, appends "(1,2,4,5)" to os.
template <typename T> void append(std::ostream &os, const std::vector<T> &t) {
  os << '(';
  if (t.size() > 0) {
    // note that std::cbegin is only introduced in C++14, so using std::begin
    // to ensure C++11 API.
    auto t0 = std::begin(t);
    os << *t0;
    ++t0;
    while (t0 != std::end(t)) {
      os << ',' << *t0;
      ++t0;
    }
  }
  os << ')';
}

extern template void append<>(std::ostream &, const std::vector<int64_t> &);
extern template void append<>(std::ostream &, const std::vector<uint64_t> &);
extern template void append<>(std::ostream &, const std::vector<int> &);
extern template void append<>(std::ostream &,
                              const std::vector<std::string> &);

} // namespace util
} // namespace poprithms

#endif
