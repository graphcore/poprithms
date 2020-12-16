// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>

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

template <typename T>
[[noreturn]] void badValue(T &&v, const std::string &type) {
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
    badValue(vals, type);
  }
}

template <typename T>
std::vector<uint64_t> get_u64(const std::vector<T> &ds) {
  std::vector<uint64_t> vs;
  vs.reserve(ds.size());
  for (auto x : ds) {
    vs.push_back(x.get());
  }
  return vs;
}
} // namespace

Dilation::Dilation(uint64_t s_) : BaseScalarU64(s_) {
  if (s_ == 0) {
    badValue(s_, "Dilation");
  }
}

Stride::Stride(uint64_t s_) : BaseScalarU64(s_) {
  if (s_ == 0) {
    badValue(s_, "Stride");
  }
}

Dilations::Dilations(const std::vector<uint64_t> &d) : BaseVectorU64(d) {
  verifyNonZero(d, "Dilations");
}

Dilations::Dilations(std::vector<uint64_t> &&d)
    : BaseVectorU64(std::move(d)) {
  verifyNonZero(d, "Dilations");
}

Dilations::Dilations(const std::vector<Dilation> &ds)
    : Dilations(get_u64(ds)) {}

Strides::Strides(const std::vector<uint64_t> &d) : BaseVectorU64(d) {
  verifyNonZero(d, "Strides");
}

Strides::Strides(std::vector<uint64_t> &&d) : BaseVectorU64(std::move(d)) {
  verifyNonZero(d, "Strides");
}

Strides::Strides(const std::vector<Stride> &ds) : Strides(get_u64(ds)) {}

} // namespace ndarray
} // namespace poprithms
