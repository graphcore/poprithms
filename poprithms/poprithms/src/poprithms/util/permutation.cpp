// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>

#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

Permutation Permutation::identity(uint64_t rnk) {
  std::vector<uint64_t> p(rnk, 0);
  std::iota(p.begin(), p.end(), 0);
  return p;
}

Permutation Permutation::prod(const std::vector<Permutation> &ps) {
  if (ps.empty()) {
    throw error("Failed to get product/composition, more than 0 "
                "Permutations required. ");
  }
  return std::accumulate(
      ps.cbegin(),
      ps.cend(),
      Permutation::identity(ps[0].size()),
      [](const Permutation &a, const Permutation &b) { return a.mul(b); });
}

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
  for (uint64_t i = 0; i < size(); ++i) {
    if (i != get(i)) {
      return false;
    }
  }
  return true;
}

std::ostream &operator<<(std::ostream &ost, const Permutation &p) {
  p.append(ost);
  return ost;
}

} // namespace util
} // namespace poprithms
