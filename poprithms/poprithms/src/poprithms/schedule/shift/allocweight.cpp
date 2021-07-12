// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <schedule/shift/error.hpp>
#include <string>

#include <boost/functional/hash.hpp>

#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class DoubleToString {
public:
  std::string operator()(double v) const {
    std::ostringstream oss;
    appendDoubleLossless(oss, v);
    return oss.str();
  }

private:
  void appendDoubleLossless(std::ostream &os, double v) const {
    if (static_cast<double>(static_cast<int>(v)) - v == 0.) {
      os << static_cast<int>(v);
    } else if (std::stod(std::to_string(v)) - v == 0.) {
      os << std::to_string(v);
    } else {
      std::ostringstream oss;
      oss << std::scientific
          << std::setprecision(std::numeric_limits<double>::digits10 + 2)
          << v;
      auto vStr = oss.str();
      if (std::stod(vStr) - v != 0.) {
        throw schedule::shift::error("Failed to serialize double " + vStr +
                                     " without being lossy");
      }
      os << vStr;
    }
  }
};

AllocWeight::AllocWeight(double _v_, int relativeLexico)
    : v{0, 0, 0, 0, 0, 0, 0} {
  //   -3 -2  -1 0  1  2  3
  if (std::abs(relativeLexico) >= (NAW + 1) / 2) {
    throw error("invalid relativeLexico in AllocWeight constructor");
  }
  v[static_cast<uint64_t>((NAW - 1) / 2 + relativeLexico)] = _v_;
}

void AllocWeight::appendSerialization(std::ostream &ost) const {
  DoubleToString d2s;
  ost << "[" << d2s(v[0]);
  for (uint64_t i = 1; i < NAW; ++i) {
    ost << ',' << d2s(v[i]);
  }
  ost << "]";
}

size_t AllocWeight::hash() const {
  size_t hash = 0u;
  boost::hash_combine(hash, v);
  return hash;
}

size_t hash_value(const AllocWeight &a) { return a.hash(); }

std::string AllocWeight::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

void AllocWeight::append(std::ostream &ost) const {
  ost << '(' << v[0];
  for (uint64_t i = 1; i < NAW; ++i) {
    ost << ',' << ' ' << v[i];
  }
  ost << ')';
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
