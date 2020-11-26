// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>

#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

Permutation::Permutation(const std::vector<uint64_t> &p_) : permutation(p_) {
  std::vector<bool> seen(permutation.size(), false);
  for (uint64_t i : permutation) {
    if (i >= permutation.size() || seen[i]) {
      std::ostringstream oss;
      oss << "Invalid permutation vector in Permutation constructor: ";
      append(oss);
      throw error(oss.str());
    }
    seen[i] = true;
  }
}

Permutation Permutation::reverse(uint64_t r) {
  std::vector<uint64_t> p_(r, 0);
  std::iota(p_.begin(), p_.end(), 0);
  std::reverse(p_.begin(), p_.end());
  return Permutation(p_);
}

std::vector<uint32_t> Permutation::get_u32() const {
  std::vector<uint32_t> out(permutation.size());
  for (uint64_t d = 0; d < size(); ++d) {
    out[d] = static_cast<uint32_t>(permutation[d]);
  }
  return out;
}

Permutation Permutation::inverse() const {
  std::vector<uint64_t> inv(size(), 0);
  for (uint64_t i = 0; i < size(); ++i) {
    inv[permutation[i]] = i;
  }
  return Permutation(inv);
}

void Permutation::append(std::ostream &ost) const {
  poprithms::util::append(ost, get());
}

void Permutation::confirmInSize(uint64_t s) const {
  if (s != size()) {
    std::ostringstream oss;
    oss << "Invalid input in Permutation::confirmInSize(" << s << ").";
    oss << ". This permutation " << *this << " is of size " << size() << '.';
    throw error(oss.str());
  }
}

bool Permutation::isIdentity() const {
  auto p2 = get();
  std::iota(p2.begin(), p2.end(), 0);
  return p2 == get();
}

std::ostream &operator<<(std::ostream &ost, const Permutation &p) {
  p.append(ost);
  return ost;
}

} // namespace util
} // namespace poprithms
