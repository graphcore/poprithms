// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_GROUPEDMATMULPACK_HPP
#define POPRITHMS_NDARRAY_GROUPEDMATMULPACK_HPP

#include <array>
#include <memory>
#include <ostream>
#include <vector>

#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace ndarray {

/**
 * A helper class for determining the shape and broadcast dimensions of a
 * grouped matmul. See
 * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
 * for the numpy broadcasting rules.
 *
 * T is a tensor class, and Molder is a class which returns shape information
 * about tensors and performs basic view-changes on tensors. See the
 * GroupedMatMulPack constructor for details.
 * */
template <typename Molder, typename T> class GroupedMatMulPack {

private:
  int64_t nGroups_;
  uint64_t M_;
  uint64_t N_;
  uint64_t K_;
  Shape outShape_ = {};
  std::unique_ptr<T> lhs3d_;
  std::unique_ptr<T> rhs3d_;

public:
  /** A rank-3 view of the lhs argument of the matmul **/
  T lhs3d() const { return *lhs3d_; }

  /** A rank-3 view of the rhs argument of the matmul **/
  T rhs3d() const { return *rhs3d_; }

  /**
   * The output shape of the matmul.  See
   * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
   * for the broadcasting rules.
   * */
  Shape outShape() const { return outShape_; }

  /**
   * The number of individual matmuls to perform in the grouped matmul.
   * */
  int64_t nGroups() const { return nGroups_; }
  uint64_t nGroups_u64() const { return static_cast<uint64_t>(nGroups_); }

  /**
   * Each of the nGroups matmuls has lhs of shape (M, K) and rhs of shape
   * (K, N).
   * */
  uint64_t M() const { return M_; }
  uint64_t N() const { return N_; }
  uint64_t K() const { return K_; }

  int64_t M_i64() const { return static_cast<int64_t>(M_); }
  int64_t N_i64() const { return static_cast<int64_t>(N_); }
  int64_t K_i64() const { return static_cast<int64_t>(K_); }

  /**
   * Construct a GroupledMatMulPack from the 2 Tensors which are to multiplied
   * together.
   * */
  GroupedMatMulPack(const T &lhs, const T &rhs) {

    const Shape lhsShape = Molder::shape(lhs);
    const Shape rhsShape = Molder::shape(rhs);
    outShape_            = lhsShape.matmul(rhsShape);

    // Increase the rank to 2, if it is 1. For both lhs (a) and rhs (b).
    auto a = lhsShape.rank_u64() == 1 ? Molder::unsqueeze(lhs, 0) : lhs;
    auto b = rhsShape.rank_u64() == 1 ? Molder::unsqueeze(rhs, 1) : rhs;

    // a is now M x K, and
    // b is now K x N.
    M_ = Molder::dim(a, a.rank_u64() - 2);
    N_ = Molder::dim(b, b.rank_u64() - 1);
    K_ = Molder::dim(a, a.rank_u64() - 1);

    const int64_t M_i64 = static_cast<int64_t>(M_);
    const int64_t N_i64 = static_cast<int64_t>(N_);
    const int64_t K_i64 = static_cast<int64_t>(K_);

    const auto aShape = Molder::shape(a).get();
    const auto bShape = Molder::shape(b).get();

    // numpy shape broadcasting, applied to all but the final 2 dimensions.
    auto preShape = Shape{{aShape.cbegin(), aShape.cend() - 2}}.numpyBinary(
        {{bShape.cbegin(), bShape.cend() - 2}});

    nGroups_ = preShape.nelms();

    auto lhs0 = Molder::expand(a, preShape.append(M_).append(K_));
    auto lhs1 = Molder::reshape(lhs0, {nGroups_, M_i64, K_i64});
    lhs3d_    = std::unique_ptr<T>(new T(lhs1));

    auto rhs0 = Molder::expand(b, preShape.append(K_).append(N_));
    auto rhs1 = Molder::reshape(rhs0, {nGroups_, K_i64, N_i64});
    rhs3d_    = std::unique_ptr<T>(new T(rhs1));
  }
};

} // namespace ndarray
} // namespace poprithms

#endif
