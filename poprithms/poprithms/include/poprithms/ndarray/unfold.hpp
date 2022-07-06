// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_UNFOLD_HPP
#define POPRITHMS_NDARRAY_UNFOLD_HPP

#include <array>
#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace ndarray {

/**
 *
 * Concatenate equally spaced slices from tensor #tIn. The slices are in
 * dimension #dim, and of width #size. The slice start positions are separated
 * by a distance #step.
 *
 * The distance between one slice ending and the next start beginning is
 * #step - #size. Note that #step - #slice may be negative. In other words,
 * the slices may overlap.
 *
 * For the case step < size (the overlapping case), we terminate unfolding
 * with the final complete unfold. For example, if
 *
 *   toUnfold = (1,2,3,4), dim=0, size=3, step=1, then the unfolded tensor is
 *   unfolded = (1,2,3,2,3,4):
 *
 *   1 2 3 4|
 *   =====  |     first unfold (1,2,3) is complete
 *     =====|     second unfold (2,3,4) is complete
 *       =====    third unfold is incomplete, and not appended to soln.
 *         =====  incomplete (ditto above).
 *          |
 *
 * For the case size >= step, incomplete slices are included. For example, if
 *
 *   toUnfold = (1,2,3,4), dim=0, size=2, step=3, then the unfolded tensor is
 *   unfolded = (1,2,4):
 *
 *   1 2 3 4|
 *   ===    |    the first unfold is complete.
 *         ===   the second unfold is incomplete. The solution contains the
 *          |    part of it within bounds.
 *          |
 *
 * \tparam T : The tensor class.
 *
 * \tparam H : The tensor manipulator, or (H)elper class. H must provide
 *             methods for slicing, broadcasting, reshaping, and concatenating
 *             a tensor (of type T). See the class TUnfoldHelper below for an
 *             example of an implemented H interface.
 *
 * \return A tensor of the same rank as #tIn, which is the same size in all
 *         dimensions except for dimension #dim. The size in dimension dim
 *         depends on #size and #step.
 * */
template <typename T, class H> class Unfolder {
public:
  static T unfold(const T &tIn, uint64_t dim, uint64_t size, uint64_t step) {

    if (step == 0) {
      throw poprithms::error::error("ndarray",
                                    "Step size in unfold cannot be 0.");
    }

    const uint64_t dimSize = H::dim(tIn, dim);

    // For the case of overlapping slices, we convert the problem into an
    // equivalent one without any overlapping slices. This is done be
    // repeating the tensor in the unfolding dimension, and increasing the
    // step. An example:
    //
    //   toUnfold=(1,2,3,4) dim=0, size=2, step=1.
    //             ===
    //               ===
    //                 ===
    //
    // Is converted to the equivalent non-overlapping problem:
    //
    //   toUnfold=(1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4) dim=0, size=2, step=5
    //             ===       ===       ===
    //
    // Note that the new, non-overlapping problem has a much larger tensor
    // (larger by a factor equal to the size of the unfold dimension #dim),
    // and so it is important that the broadcast is just a view-change, and
    // not a new variable.
    //
    if (step < size) {
      auto ub = size * (1 + (dimSize - size) / step);
      auto t  = tIn;
      t       = unsqueeze(t, dim);
      t       = H::broadcast(t, dimSize, dim);
      t       = flatten(t, dim, dim + 2);

      // solve the equivalent non-overlapping problem (depth-1 recursion).
      t = Unfolder<T, H>::unfold(t, dim, size, step + dimSize);
      t = H::slice(t, dim, 0, ub);
      return t;
    }

    // The non-overlapping case.
    else {

      // The number of complete stripes:
      const uint64_t nStripes = dimSize / step;

      std::vector<T> toConcat;
      toConcat.reserve(2);

      if (nStripes > 0) {
        // Gather up the complete stripes:
        auto t = tIn;
        t      = H::slice(t, dim, 0, nStripes * step);
        t      = reshapePartial(t, dim, dim + 1, {nStripes, step});
        t      = H::slice(t, dim + 1, 0, size);
        t      = flatten(t, dim, dim + 2);
        toConcat.push_back(t);
      }

      // Get the remaining elements, if there are any:
      const auto s0 = nStripes * step;
      const auto s1 = std::min<uint64_t>(s0 + size, dimSize);
      if (s0 != s1) {
        toConcat.push_back(H::slice(tIn, dim, s0, s1));
      }

      if (toConcat.size() == 1) {
        return toConcat[0];
      }
      return H::concat(toConcat, dim);
    }
  }

private:
  static Shape shape(const T &t) { return Shape::createFrom(H::shape(t)); }

  static T flatten(const T &t, uint64_t dim0, uint64_t dim1) {
    return H::reshape(t, shape(t).flatten(dim0, dim1).get_u64());
  }

  static T unsqueeze(const T &t, uint64_t d) {
    return H::reshape(t, shape(t).unsqueeze(d).get_u64());
  }

  static T reshapePartial(const T &t,
                          uint64_t dim0,
                          uint64_t dim1,
                          const std::vector<uint64_t> &newDims) {
    std::vector<int64_t> dims{newDims.cbegin(), newDims.cend()};
    return H::reshape(t, shape(t).reshapePartial(dim0, dim1, dims).get_u64());
  }
};

/**
 * Templatized unfold helper class for tensors with APIs like the
 * poprithms::compute::host::Tensor.
 * */
template <typename T> class TUnfoldHelper {

public:
  static T slice(const T &t, uint64_t dim, uint64_t start, uint64_t end) {
    return t.slice_(Dimension(dim), start, end);
  }

  static T broadcast(const T &t, uint64_t N, uint64_t dim) {
    return t.expand_(t.shape().broadcast(N, dim));
  }

  static T reshape(const T &t, const std::vector<uint64_t> &shape) {
    return t.reshape_(Shape::createFrom(shape));
  }

  static T concat(const std::vector<T> &ts, uint64_t axis) {
    return T::concat_(ts, axis);
  }

  static uint64_t dim(const T &t, uint64_t d) { return t.dim(d); }

  static std::vector<uint64_t> shape(const T &t) {
    return t.shape().get_u64();
  }
};

} // namespace ndarray
} // namespace poprithms

#endif
