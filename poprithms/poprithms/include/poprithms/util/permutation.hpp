// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_PERMUTATION_HPP
#define POPRITHMS_UTIL_PERMUTATION_HPP

#include <ostream>
#include <vector>

namespace poprithms {
namespace util {

/**
 * A class to represent a permutation.
 * */
class Permutation {

public:
  /**
   * \param p_ A vector of n distinct values in the range [0, n) which defines
   *           the Permutation.
   * */
  Permutation(const std::vector<uint64_t> &p_);

  /**
   * \return The identity permutation, (0 1 2 ... rnk-1).
   * */
  static Permutation identity(uint64_t rnk);

  /**
   * A pair of indices for a dimesion roll.
   * */
  struct DimRollPair {
    template <typename T>
    DimRollPair(T f, T t)
        : from_(static_cast<uint64_t>(f)), to_(static_cast<uint64_t>(t)) {}
    uint64_t from() const { return from_; }
    uint64_t to() const { return to_; }

  private:
    uint64_t from_;
    uint64_t to_;
  };

  /**
   * A special kind of Permutation, where one dimension migrates, and all
   * other dimensions retain their relative order.
   * */
  static Permutation dimRoll(uint64_t rnk, DimRollPair p);

  /**
   * \return true if and only if (iff) this Permutation is
   *         (0 1 2 ... size() -1)
   * */
  bool isIdentity() const;

  /**
   * \return A Permutation which reverses the order of dimensions.
   *
   * \param r The rank of the Permutation.
   * */
  static Permutation reverse(uint64_t r);

  /**
   * \return a Permutation which permutes the final 2 dimensions. This is
   *         sometimes called the 'transpose' Permutation.
   * */
  static Permutation reverseFinalTwo(uint64_t r);

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

  /**
   * Multiply/compose this permutation on the right-hand side by \a rhs
   * Example (1 2 3 0).mul((1 2 3 0)) is (2 3 0 1).
   * https://en.wikipedia.org/wiki/Permutation_group
   *  */
  Permutation mul(const Permutation &rhs) const { return apply(rhs.get()); }

  /** Multiply this Permutation by itself \a p times. */
  Permutation pow(int64_t p) const;

  /**
   * Accumulate the Permutations in \a prms using multiplication, also known
   * as composition.
   * */
  static Permutation prod(const std::vector<Permutation> &prms);

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
   * Suppose this permutation is (1 2 0). Then
   *   mapForward({0})     is {2}
   *   mapForward({0,1})   is {2,0}
   *   mapForward({0,1,2}) is {2,0,1}.
   * */
  std::vector<uint64_t>
  mapForward(const std::vector<uint64_t> &indicesBefore) const;

  /**
   * Suppose this permutation is (1 2 0). Then
   *   mapBackward({0})     is {1}
   *   mapBackward({0,1})   is {1,2}
   *   mapBackward({0,1,2}) is {1,2,0}.
   * */
  std::vector<uint64_t>
  mapBackward(const std::vector<uint64_t> &indicesAfter) const;

  /**
   * Consider permuting and then subsampling a vector v. In particular,
   * consider the sequence of operations:
   *
   *   1) permute v
   *   2) find where the elements with starting positions `where` ended up
   *   3) select them, retaining their new order.
   *
   * For example suppose the permutation is
   *     permutation = (4 2 5 1 3 0),
   *
   * that the indices to select are
   *     where = (0,3,4),
   *
   * and that the input vector is
   *     v = ("a", "b", "c", "d", "e", "f").
   *
   * 1) The permutation of v is ("e", "c", "f", "b", "d", "a").
   * 2) we see v[0] = "a" went to index 5
   *           v[3] = "d" went to index 4
   *           v[4] = "e" when to index 0.
   *
   * 3) select "a", "d" and "e" in their new order:("e", "d", "a").
   *
   *
   * It is possible to do this in a different order and end up with the same
   * result. In particular,
   *
   *    1) select the elements at positions `where` in v
   *    2) permute with subPermutation.
   *
   * What is subPermutation? That is exactly what this class method
   * determines.
   *
   * In the above example it must be (2 1 0) to
   * arrive at the same ("e", "d", "e") which we obtained with the first
   * method.
   *
   *
   * It can be shown that the exact relationship which subPermutation must
   * have with permutation is:
   *
   * subPermutation.apply(v.at(where))
   *                      ==
   *                permutation.apply(v).at(permutation.inverse().at(where)).
   *
   *
   * Some more examples:
   *
   *
   * If this Permutation is (1 2 0) and where is (0,2) :
   *               = =
   * then return (1 0)
   *
   * If this Permutation is (2 1 0) and where is (0,2) :
   *             =   =
   * then return (1 0).
   *
   * If this Permutation is (4 2 5 1 3 0) and where is (0,3,4)
   *                         =       = =
   * then return (2 1 0).
   *
   *
   * If this is (2 1 3 0) and where is (0,2,3):
   *             =   = =
   * then return  (1 2 0).
   *
   *  If this is (4 6 0 5 2 1 3) and where is (2,3,5,6)
   *              =   = =   =
   *  then return  (3 2 0 1).
   *
   *
   *
   *  \a where does not need to be in order, but the elements must be unique;
   *
   * Case where the elements in \a where are not in ascending order.
   * If this is (1 2 0) and where is (1,0):
   *             =   =
   * then return (0 1).
   *
   * */
  Permutation subPermutation(const std::vector<uint64_t> &where) const;

  /**
   * Return true if the sequence \a query is contained as a contiguous
   * subsequence of this Permutation.
   *
   * Example if this is (2 0 3 1) and query is (0,3,1), return true
   * Example if this is (2 0 3 1) and query is (2,0,1), return false
   *
   * */
  bool containsSubSequence(const std::vector<uint64_t> &query) const;

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
