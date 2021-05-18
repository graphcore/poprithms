// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <ostream>

#include <poprithms/ndarray/accessors.hpp>
#include <poprithms/ndarray/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace ndarray {

template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &vs) {
  util::append(ost, vs);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Starts &starts) {
  ost << starts.vals;
  return ost;
}

template <typename T>
[[noreturn]] void throwBadValue(T &&v, const std::string &type) {
  std::ostringstream oss;
  oss << "Invalid " << type << ": " << v;
  oss << ". " << type << " must be strictly positive (0 not allowed). ";
  throw error(oss.str());
}

namespace {
template <typename T>
void verifyNonZero(const std::vector<T> &vals, const std::string &type) {
  if (std::any_of(
          vals.cbegin(), vals.cend(), [](auto x) { return x == 0; })) {
    throwBadValue(vals, type);
  }
}
} // namespace

Dilation::Dilation(uint64_t s_) : BaseScalarU64(s_) {
  if (s_ == 0) {
    throwBadValue(s_, "Dilation");
  }
}

Stride::Stride(uint64_t s_) : BaseScalarU64(s_) {
  if (s_ == 0) {
    throwBadValue(s_, "Stride");
  }
}

std::ostream &operator<<(std::ostream &o, const Dilations &s) {
  util::append(o, s.get());
  return o;
}

std::ostream &operator<<(std::ostream &o, const Strides &s) {
  util::append(o, s.get());
  return o;
}

std::ostream &operator<<(std::ostream &o, const Dimensions &s) {
  util::append(o, s.get());
  return o;
}

std::ostream &operator<<(std::ostream &ost,
                         const std::vector<Dimensions> &dimss) {
  util::append(ost, dimss);
  return ost;
}

} // namespace ndarray
} // namespace poprithms
