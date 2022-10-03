// Copyrigh_ (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_STRIDEANDOFFSET_HPP
#define POPRITHMS_NDARRAY_STRIDEANDOFFSET_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <sstream>
#include <tuple>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/settutil.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace ndarray {

/**
 *
 * \tparam T : The tensor class.
 *
 * \tparam H : The tensor manipulator, or (H)elper class. H must provide
 *             methods for slicing, dimShuffling, concatenating, reversing and
 *             reshaping a tensor (of type T). See the class
 *             TFromStrideAndOffsetHelper below for an example of an
 *             implemented H interface.
 **/
template <typename T, class H> class FromStrideAndOffset {
public:
  /**
   * Translate from the ptr-offet-strides representation into a DAG of
   * view-changing operations.
   *
   * What is ptr-offset-strides representation? In numpy, arrays (tensors) are
   * stored in the ptr-offset-strides form, something like
   *
   * <code>
   * struct TensorImpl {
   *   float * alloc;
   *   int64_t offset;
   *   std::vector<int64_t> strides;
   *   std::vector<int64_t> shape;
   * };
   * </code>
   *
   * The above ptr-offset-strides format encodes a strided view into the
   * contiguous chunk of memory starting at #alloc. The tensor encoded has
   * shape #shape and the element at index (i_0, i_1, ... ,i_{rank-1})
   * is alloc0[offset + sum_{all dimensions j < rank} (i_j * strides[j])].
   *
   * Example 1:
   * <code>
   * > in0 = np.ones((4,4),dtype=np.bool)
   * > b   = a[1:3,2:4]
   * </code>
   *
   * #in0 is allocated a new contiguous chunk of 16 elements in memory, and b
   * is a view into #in0 with b.strides = (4,1) and offset of 4*1 + 2*1 = 6.
   *
   * Example 2:
   * <code>
   * > in0 = np.arange(9).reshape((3,3))
   * > b   = a[2::-2, 1].T
   * </code>
   *
   * b.strides is (-48,) (in bytes, so strides is (-6,) per element) and the
   * offset of b is 14 (elements).
   *
   * \param in0 The input tensor. This is the equivalent of #alloc0 in the
   *            above description.
   *
   * \param strides0 The number of elements separating consecutive elements in
   *        each dimension. This is the equivalent of #strides in the
   *        description above.
   *
   * \param offset The number of elements between the first element of #in0
   *        and the first element of the view into it.
   *
   * \param outShape The output shape.
   *
   * \return A view into #in0.
   * */
  static T asStrided(const T &in0,
                     const std::vector<int64_t> &outStrides,
                     int64_t outOffset,
                     const Shape &outShape) {

    if (outStrides.size() != outShape.rank_u64()) {
      std::ostringstream oss;
      oss << "Shape and strides have different ranks. "
          << "Shape has rank " << outShape.rank_u64()
          << " and strides has rank " << outStrides.size() << ".";
      throw poprithms::error::error("ndarray", oss.str());
    }

    if (in0.nelms() < outShape.nelms()) {
      std::ostringstream oss;
      oss << "The input tensor (the data which a view is being taken of) has "
          << in0.nelms() << ". The output tensor (the view) has shape "
          << outShape.nelms() << ", and therefore " << outShape.nelms()
          << " elements. " << in0.nelms() << " < " << outShape.nelms()
          << " (not allowed).";
      throw poprithms::error::error("ndarray", oss.str());
    }

    // Edge case: when the output tensor has zero elements.
    if (outShape.nelms() == 0) {
      if (in0.nelms() == 0) {
        return H::reshape(in0, outShape);
      }
      return H::slice(H::flatten(in0), Dimension(0), 0, 0);
    }

    // Edge case: when the output tensor has 1 element.
    if (outShape.nelms() == 1) {
      return H::reshape(
          H::slice(H::flatten(in0), Dimension(0), outOffset, outOffset + 1),
          outShape);
    }

    // Remove then singleton dimensions (will be reinserted before returning).
    auto nonSingletonDims = outShape.nonSingletonDimensions();

    std::vector<int64_t> vSqueezedOutShape;
    vSqueezedOutShape.reserve(nonSingletonDims.size());

    std::vector<int64_t> squeezedOutStrides;
    squeezedOutStrides.reserve(nonSingletonDims.size());

    for (auto d : nonSingletonDims) {
      vSqueezedOutShape.push_back(outShape.dim(d));
      squeezedOutStrides.push_back(outStrides[d]);
    }
    Shape squeezedOutShape(vSqueezedOutShape);

    // The shape of in0 is not required, in0 is used only as the "ptr" in
    // numpy's ptr-offset-strides format. We can flatten it.
    auto in0flat = H::flatten(in0);

    // The chain of view-changing ops applied to in0flat will be:
    //
    //    in -> slice -> settsample -> dimshuffle -> reverse.
    //
    // where a settsample is a generalization of strided slice.
    //
    // We first compute the parameters for the 4 operations, roughly in the
    // reverse order they will be applied:
    // (1) get reverse indices and offset of slice
    // (2) get dimshuffle permutation
    // (3) get settsample (generalized) slice bounds
    //
    // and the apply the operations
    // (4) slice, using offset computed in (1)
    // (5) settsample, using bounds computed in (3)
    // (6) dimshuffle, using permutation computed in (2)
    // (7) reverse, using indices computed in (1)
    //
    //
    //
    // We start by computing reverse dimensions and the offset before the
    // reverse is applied.
    //
    // in -> slice -> settsample -> dimshuffle -> reverse
    //                                            =======
    auto offsetPreReverse = outOffset;
    std::vector<uint64_t> revDims;
    for (uint64_t i = 0; i < squeezedOutStrides.size(); ++i) {
      if (squeezedOutStrides[i] < 0) {
        revDims.push_back(i);
        offsetPreReverse +=
            squeezedOutStrides[i] * (squeezedOutShape.dim(i) - 1);
      }
    }

    // Just before the reverse, there is dimshuffle. Below, we compute the
    // permutation required.
    //
    // in -> slice -> settsample -> dimshuffle -> reverse
    //                              ==========
    //
    std::vector<std::pair<int64_t, uint64_t>> absStrideAndDim;
    for (uint64_t i = 0; i < squeezedOutStrides.size(); ++i) {
      int64_t stride    = squeezedOutStrides[i];
      int64_t absStride = std::abs(stride);
      absStrideAndDim.push_back({absStride, i});
    }
    std::sort(absStrideAndDim.begin(),
              absStrideAndDim.end(),
              std::greater<std::pair<int64_t, uint64_t>>());
    std::vector<uint64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> vShape;
    for (auto &&p : absStrideAndDim) {
      auto dim       = p.second;
      auto absStride = p.first;
      dims.push_back(dim);
      vShape.push_back(squeezedOutShape.dim(dim));
      strides.push_back(absStride);
    }
    auto p = Permutation(dims).inverse();

    // The shape before the dimshuffle:
    Shape shape(vShape);

    // Just before the dimshuffle, there is settsample (generalized slice).
    // Below, we compute the setts (generalized slice bounds).
    //
    // in -> slice -> settsample -> dimshuffle -> reverse
    //                ==========
    using poprithms::memory::nest::Stripe;
    std::vector<Stripe> stripes;
    const auto N = strides.size();
    for (uint64_t i = 0; i < N; ++i) {
      int64_t stridePrevDim = i == 0 ? in0.nelms_u64() : strides.at(i - 1);
      int64_t on =
          std::min<int64_t>(stridePrevDim, shape.dim(i) * strides.at(i));
      int64_t off = stridePrevDim - on;
      Stripe s0{on, off, 0};
      stripes.push_back(s0);
    }

    if (strides.empty()) {
      throw poprithms::error::error("ndarray",
                                    "Empty strides, but output has more than "
                                    "1 element. Internal logic error?");
    }
    stripes.push_back({1, strides.back() - 1, 0});

    // We are ready to apply the chain of operations. We start with the slice
    // (handles the offset):
    //
    // (4) in -> slice -> settsample -> dimshuffle -> reverse
    //           =====
    auto slicedFlatIn = H::slice(
        in0flat, Dimension(0), offsetPreReverse, H::nelms_u64(in0flat));

    // (5) in -> slice -> settsample -> dimshuffle -> reverse
    //                    ==========
    poprithms::memory::nest::Region region({slicedFlatIn.nelms()}, {stripes});

    // This utility class implements a settsample in terms of the basic view
    // changing operations that poplar supports (slice, concat, et cetera).
    poprithms::memory::chain::NonNativeSettSampler nnsa;
    auto sampled       = nnsa.settSample<T, H>(slicedFlatIn, region);
    auto preDimShuffle = H::reshape(sampled, shape);

    // (6) in -> slice -> settsample -> dimshuffle -> reverse
    //                                  ==========
    auto preReverse = H::dimShuffle(preDimShuffle, p);

    // (7) in -> slice -> settsample -> dimshuffle -> reverse
    //                                                =======
    auto out = H::reverse(preReverse, revDims);

    // reinsert singeton dimensions and return.
    return out.reshape(outShape);
  }
};

/**
 * Templatized helper class for tensors with APIs like the
 * poprithms::compute::host::Tensor.
 * */
template <typename T> class TFromStrideAndOffsetHelper {
public:
  static Shape shape(const T &t) { return t.shape(); }

  static uint64_t nelms_u64(const T &t) { return t.nelms_u64(); }

  static T slice(const T &t, Dimension d, uint64_t l, uint64_t u) {
    return t.slice(d, l, u);
  }
  static T reshape(const T &t, const Shape &s) { return t.reshape(s); }

  static T concat(const std::vector<T> &ts, uint64_t d) {
    return T::concat(ts, d);
  }
  static T flatten(const T &t) { return t.flatten(); }

  static T dimShuffle(const T &t, const Permutation &p) {
    return t.dimShuffle(p);
  }

  static T reverse(const T &t, const std::vector<uint64_t> &dims) {
    return t.reverse(dims);
  }
};

} // namespace ndarray
} // namespace poprithms

#endif
