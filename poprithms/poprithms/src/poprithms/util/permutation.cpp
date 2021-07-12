// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>

#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

bool Permutation::containsSubSequence(
    const std::vector<uint64_t> &query) const {
  if (query.empty()) {
    return true;
  }

  // Find the unique point in this Permutation which is the same as the start
  // of the query, if the start of the query appears on this Permutation at
  // all.
  auto found = std::find(permutation.cbegin(), permutation.cend(), query[0]);

  // If the distance to the end from the found position is less than the query
  // length, then it's impossible for the query to be in this permutation
  if (std::distance(found, permutation.cend()) <
      static_cast<int64_t>(query.size())) {
    return false;
  }

  // Check for the match
  return std::vector<uint64_t>(found, std::next(found, query.size())) ==
         query;
}

Permutation
Permutation::subPermutation(const std::vector<uint64_t> &where) const {

  for (auto w : where) {
    if (w >= size()) {
      std::ostringstream oss;
      oss << "Invalid element of `where` in subPermutation for "
          << "this Permutation, " << *this << ". The element " << w
          << " is too large to index into this Permutation, "
          << "which is only of rank " << size() << '.';
      throw error(oss.str());
    }
  }

  //  this   (4 2 5 1 3 0)
  //          =   =     =
  //  where  (0,4,5)
  //
  //  inv    (5 3 1 4 0 2)
  //          ^       ^ ^
  //          |       | |
  //          0       4 5
  const auto inv = inverse();

  constexpr int64_t blank{-1};
  std::vector<int64_t> mapping(size(), blank);
  for (uint64_t i = 0; i < where.size(); ++i) {
    // mapping[5] = 0
    // mapping[0] = 1
    // mapping[2] = 2
    mapping[inv.get(where[i])] = i;
  }

  std::vector<uint64_t> subPerm;
  subPerm.reserve(where.size());
  for (auto x : mapping) {
    if (x != blank) {
      subPerm.push_back(x);
    }
  }

  return subPerm;
}

Permutation Permutation::dimRoll(uint64_t rnk, DimRollPair p) {
  if (p.from() >= rnk) {
    std::ostringstream oss;
    oss << "Invalid source of dimRoll, " << p.from()
        << ". Source must be less than rank, " << rnk << '.';
    throw error(oss.str());
  }

  if (p.to() >= rnk) {
    std::ostringstream oss;
    oss << "Invalid destination of dimRoll, " << p.to()
        << ". Destination must be less than rank, " << rnk << '.';
    throw error(oss.str());
  }

  auto perm = identity(rnk).get();

  const int64_t fwd = p.from() < p.to() ? +1 : -1;
  for (auto i = p.from(); i != p.to(); i += fwd) {
    perm[i] += fwd;
  }
  perm[p.to()] = p.from();
  return Permutation(perm);
}

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

Permutation Permutation::reverseFinalTwo(uint64_t r) {
  if (r < 2) {
    throw error("Cannot create permutation of 'final 2' dimensions which " +
                std::string("is only of rank ") + std::to_string(r) +
                ". Rank must be at least 2.");
  }
  std::vector<uint64_t> p_(r, 0);
  std::iota(p_.begin(), p_.end() - 2, 0);
  p_[r - 2] = r - 1;
  p_[r - 1] = r - 2;
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

std::vector<uint64_t>
Permutation::mapBackward(const std::vector<uint64_t> &indicesAfter) const {

  for (auto i : indicesAfter) {
    if (i >= size()) {
      std::ostringstream oss;
      oss << "Invalid value in mapBackwards, " << i
          << ", for this Permutation which is only of size  " << size()
          << '.';
      throw error(oss.str());
    }
  }
  std::vector<uint64_t> indicesBefore;
  indicesBefore.reserve(indicesAfter.size());
  for (auto i : indicesAfter) {
    indicesBefore.push_back(permutation[i]);
  }
  return indicesBefore;
}

std::vector<uint64_t>
Permutation::mapForward(const std::vector<uint64_t> &indicesBefore) const {
  return inverse().mapBackward(indicesBefore);
}

std::ostream &operator<<(std::ostream &ost, const Permutation &p) {
  p.append(ost);
  return ost;
}

Permutation Permutation::pow(int64_t p) const {
  p = p % size();
  p += size();
  p %= size();

  Permutation s = Permutation::identity(size());
  for (int64_t i = 0; i < p; ++i) {
    s = s.mul(*this);
  }
  return s;
}

} // namespace util
} // namespace poprithms
