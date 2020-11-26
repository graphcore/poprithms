// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_PERMUTATION_HPP
#define POPRITHMS_UTIL_PERMUTATION_HPP

#include <ostream>
#include <vector>

namespace poprithms {
namespace util {

/**
 * A simple class to represent a permutation.
 * */
class Permutation {

public:
  /**
   * \param p_ A vector of n distinct values in the range [0, n) which defines
   *           the Permutation.
   * */
  Permutation(const std::vector<uint64_t> &p_);

  /**
   * \return A Permutation which reverses the order of dimensions.
   *
   * \param r The rank of the Permutation.
   * */
  static Permutation reverse(uint64_t r);

  const std::vector<uint64_t> &get() const { return permutation; }
  std::vector<uint32_t> get_u32() const;
  uint64_t get(uint64_t d) const { return permutation[d]; }

  /**
   * \return  The inverse Permutation. In particular the following is true:
   *          apply(inverse().get()).isIdentity().
   * */
  Permutation inverse() const;

  void append(std::ostream &) const;

  uint64_t size() const { return permutation.size(); }

  template <typename T> std::vector<T> apply(const std::vector<T> &x) const {

    // The input must be the size of this Permutation
    confirmInSize(x.size());
    std::vector<T> permuted;
    permuted.reserve(x.size());
    for (uint64_t d = 0; d < size(); ++d) {
      permuted.push_back(x[permutation[d]]);
    }
    return permuted;
  }

  /**
   * \return true iff permutation is {0, ..., size() - 1}.
   * */
  bool isIdentity() const;

  bool operator==(const Permutation &rhs) const {
    return permutation == rhs.permutation;
  }
  bool operator!=(const Permutation &rhs) const { return !operator==(rhs); }

private:
  std::vector<uint64_t> permutation;
  void confirmInSize(uint64_t) const;
};

std::ostream &operator<<(std::ostream &, const Permutation &);

} // namespace util
} // namespace poprithms

#endif
