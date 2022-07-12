// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_UNFOLD_HPP
#define POPRITHMS_NDARRAY_UNFOLD_HPP

#include <array>
#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace ndarray {

template <typename T, class H> class Unfolder {
public:
  /**
   * An unfold operation following the specification of PyTorch:
   *
   * https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
   *
   * Concatenate equally spaced slices from tensor #tIn. The slices are in
   * dimension #dim, and of width #size. The slice start positions are
   * separated by a distance #step.
   *
   * The distance between one slice ending and the next one start beginning is
   * #step - #size. Note that #step - #slice may be negative. In other words,
   * the slices may overlap.
   *
   * Incomplete slices at the end of the slice range are not included.
   *
   * \tparam T : The tensor class.
   *
   * \tparam H : The tensor manipulator, or (H)elper class. H must provide
   *             methods for slicing, broadcasting, concatenating, and
   *             reshaping a tensor (of type T). See the class TUnfoldHelper
   *             below for an example of an implemented H interface.
   *
   * \return A tensor that has a rank which is 1 greater than that of #tIn.
   *
   * Shape example:
   *
   * If #tIn has shape
   *     (s0, s1, s2, ...., sZ) and dim=1,
   *
   * then the returned tensor has shape
   *     (s0, nSlices, s2,...., sZ, size) where
   *
   * nSlices = (s1 - size) / step + 1.
   * */
  static T unfold(const T &tIn, uint64_t dim, uint64_t size, uint64_t step) {

    if (step == 0) {
      throw poprithms::error::error("ndarray",
                                    "Step size in unfold cannot be 0.");
    }

    // The size of the dimension being unfolded.
    const uint64_t dimSize = H::dim(tIn, dim);

    // The total number of complete slices which can be obtained from the
    // dimension.
    const auto nSlices = size > dimSize ? 0 : 1 + (dimSize - size) / step;

    // The shape of the result tensor.
    Shape outShape = shape(tIn).append(size).resizeSingleDim(nSlices, dim);

    // PyTorch throws an error if size > dimSize, we do not.
    if (size > dimSize || size == 0) {
      return H::reshape(H::slice(tIn, dim, 0, 0), outShape.get_u64());
    }

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
      auto t = tIn;
      t      = unsqueeze(t, dim);
      t      = H::broadcast(t, dimSize, dim);
      t      = flatten(t, dim, dim + 2);

      // solve the equivalent non-overlapping problem (depth-1 recursion).
      t = Unfolder<T, H>::unfold(t, dim, size, step + dimSize);
      t = H::slice(t, dim, 0, nSlices);
      return t;

    }

    // The non-overlapping case.
    else {

      // The number of complete stripes:
      const uint64_t nCompleteSteps = dimSize / step;

      std::vector<T> toConcat;
      toConcat.reserve(2);

      if (nCompleteSteps > 0) {
        // Gather up the complete steps:
        auto t = tIn;
        t      = H::slice(t, dim, 0, nCompleteSteps * step);
        t      = reshapePartial(t, dim, dim + 1, {nCompleteSteps, step});
        t      = H::slice(t, dim + 1, 0, size);
        toConcat.push_back(t);
      }

      // Get the remaining elements, if they form a complete stripe:
      const auto s0 = nCompleteSteps * step;
      if (s0 + size <= dimSize) {
        toConcat.push_back(unsqueeze(H::slice(tIn, dim, s0, s0 + size), dim));
      }

      auto concatted = [&]() {
        if (toConcat.size() == 1) {
          return toConcat[0];
        }
        return H::concat(toConcat, dim);
      }();

      return dimRoll(concatted, dim + 1, H::rank_u64(concatted) - 1);
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

  static T dimRoll(const T &t, uint64_t from, uint64_t to) {
    const auto r = H::rank_u64(t);
    const auto p = Permutation::dimRoll(r, {from, to});
    return H::dimShuffle(t, p.get());
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

  static T dimShuffle(const T &t, const std::vector<uint64_t> &permutation) {
    return t.dimShuffle_({permutation});
  }

  static uint64_t dim(const T &t, uint64_t d) { return t.dim(d); }

  static uint64_t rank_u64(const T &t) { return t.rank_u64(); }

  static std::vector<uint64_t> shape(const T &t) {
    return t.shape().get_u64();
  }
};

} // namespace ndarray
} // namespace poprithms

#endif
