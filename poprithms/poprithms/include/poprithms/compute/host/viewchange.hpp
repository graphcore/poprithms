// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_VIEWCHANGE_HPP
#define POPRITHMS_COMPUTE_HOST_VIEWCHANGE_HPP

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <sstream>

#include <poprithms/compute/host/usings.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace compute {
namespace host {

/** A class to help the ViewChange<Number> class with functionality which does
 * not depend on its template parameter Number. */
class ViewChangeHelper {
public:
  /** Throw a descriptive error if \a nPtrs and \a nShapes are different */
  static void assertConcatSizes(uint64_t nPtrs, uint64_t nShapes);

  /** Throw a descriptive error if \a from.expand(to) and \a to are different
   */
  static void assertExpandableTo(const Shape &from, const Shape &to);

  /** Throw a descriptive error if \a observed and \a expected are different
   */
  static void assertExpandedNElms(uint64_t observed, uint64_t expected);
  static Shape prePadToRank(const Shape &a, uint64_t r);

  /** Return a schedule for traversing \a from and \a from.dimShuffle(p) in a
   * tiled fashion. This is important for cache locality. */
  struct OldNew {
    /** A row-major index in \a from */
    uint64_t o;
    /** The corresponding row-major index in \a from.dimShuffle(p) */
    uint64_t n;
  };
  static std::vector<OldNew> getTiled(const Shape &from,
                                      const Permutation &p);

  /** Throw an error stating that ViewChange::Data does not accept nullptr
   * arguments. */
  [[noreturn]] static void nullptrDataNotAllowed();
};

/** A class to perform rearrangements of row-major arrays of data.
 * */
template <typename Number> class ViewChange {

public:
  /** A wrapper struct for a Shape \a shape, and a pointer \a data to
   * the first element of a const array. The value of data[i] must be the i'th
   * row major value in shape.  */
  struct Data {
    Shape shape;
    const Number *data;

    Data(const Shape &shape_, const Number *data_)
        : shape(shape_), data(data_) {
      // We check that data_ is not nullptr, but we cannot check that data_ is
      // indeed the first element in a contiguous array of the required
      // size. It is the users responsibility to ensure this.
      assertNotNullptr(data_);
    }

    Data(Shape &&shape_, const Number *data_)
        : shape(std::move(shape_)), data(data_) {
      assertNotNullptr(data_);
    }
  };

  static void assertNotNullptr(const Number *data_) {
    if (!data_) {
      ViewChangeHelper::nullptrDataNotAllowed();
    }
  }

  /** \return values in row-major order of the expansion of \a input to Shape
   *          \a to */
  static std::vector<Number> expand(const Data &input, const Shape &to) {
    ViewChangeHelper::assertExpandableTo(input.shape, to);

    // prepend 1's to the input shape, up to the rank of the target shape
    const auto from =
        ViewChangeHelper::prePadToRank(input.shape, to.rank_u64());
    const auto N0 = input.shape.nelms_u64();

    // initialize the output to the input,
    std::vector<Number> expanded(N0);
    std::copy(input.data, std::next(input.data, N0), expanded.begin());

    // incrementally expand along dimemions of different sizes
    auto shp = from.get();
    for (uint64_t d_ = from.rank_u64(); d_ != 0; --d_) {
      const auto d = d_ - 1;
      if (from.dim(d) != to.dim(d)) {

        // Expand in dimension d
        expanded = expandSingleDim(expanded, {shp}, d, to.dim_u64(d));
        shp[d]   = to.dim(d);
      }
    }

    // Assert that the number of elements in the final expanded vector matches
    // the number of elements in the output Shape.
    ViewChangeHelper::assertExpandedNElms(expanded.size(), to.nelms_u64());
    return expanded;
  }

  /** \return values in row-major order obtained by permutating \a input with
   *          Permutation \a p */
  static std::vector<Number> dimShuffle(const Data &input,
                                        const Permutation &p) {

    // Use blocking/tiling, to improve cache hits. This object is a list of
    // (input, output) indices, ordered in such that a way that the input and
    // output arrays are accessed tile by tile. This can be thought of as
    // "tile-major" order.
    const auto blockedOldAndNew = ViewChangeHelper::getTiled(input.shape, p);
    std::vector<Number> out(input.shape.nelms_u64());

    std::for_each(blockedOldAndNew.cbegin(),
                  blockedOldAndNew.cend(),
                  [&out, &input](ViewChangeHelper::OldNew oAndN) {
                    out[oAndN.n] = input.data[oAndN.o];
                  });
    return out;
  }

  static std::vector<Number> reverse(const Data &input,
                                     const std::vector<uint64_t> &dims) {
    return fromIndices(input, input.shape.getReversedRowMajorIndices(dims));
  }

  static std::vector<Number> subSample(const Data &input,
                                       const std::vector<uint64_t> &strides) {
    return fromIndices(input,
                       input.shape.getSubSampledRowMajorIndices(strides));
  }

  /** \return values in row-major order obtained by slicing \a input between
   *          the bounds \a u and \a l. */
  static std::vector<Number>
  slice(const Data &input, const Lower &l, const Upper &u) {
    return fromIndices(input, input.shape.getSlicedRowMajorIndices(l, u));
  }

  static std::vector<Number> slice(const Data &input,
                                   const NormalizedSliceParams &n) {
    return fromIndices(input, input.shape.getSlicedRowMajorIndices(n));
  }

  /** \return values in row-major order obtained by numpy-slicing \a input
   *          between \a starts and \a ends with step sizes \a steps, in
   *          dimensions \a dims. Negative indices are fully supported, see
   *          https://numpy.org/doc/stable/reference/arrays.indexing.html
   * */
  static std::vector<Number> slice(const Data &input,
                                   const Starts &starts,
                                   const Ends &ends,
                                   const Steps &steps,
                                   const Dims &dims) {
    return slice(
        input,
        input.shape.getNormalizedSliceParams(starts, ends, steps, dims));
  }

  /** \return values in row-major order obtained by gathering and
   *          concatenating all slices in dimension \a dimension at indices \a
   *          where */
  static std::vector<Number> gather(const Data &input,
                                    uint64_t dimension,
                                    const std::vector<int64_t> &where) {
    return fromIndices(input,
                       input.shape.gatherRowMajorIndices(dimension, where));
  }

  static std::vector<Number>
  scatterToZero(const Data &input,
                const Shape &outShape,
                const std::vector<std::vector<int64_t>> &where) {
    const auto indices = outShape.gatherRowMajorIndices(where);
    std::vector<Number> out(outShape.nelms(), Number(0));
    for (uint64_t i = 0; i < indices.size(); ++i) {
      out[indices[i]] = input.data[i];
    }
    return out;
  }

  /** \return values in row-major order obtained by concatenating arrays with
   *          Shapes \a shapes and pointers \a ts, along dimension axis. */
  static std::vector<Number> concat(const std::vector<const Number *> &ts,
                                    const Shapes &shapes,
                                    const uint64_t axis) {

    ViewChangeHelper::assertConcatSizes(ts.size(), shapes.size());
    const auto outShape      = Shape::concat(shapes, axis);
    const auto concatSources = Shape::getRowMajorConcatSources(shapes, axis);
    std::vector<Number> concatted(outShape.nelms_u64());
    for (uint64_t i = 0; i < outShape.nelms_u64(); ++i) {
      const auto sourceIndex = concatSources[i].sourceShapeIndex;
      const auto sourceRowMajorIndex =
          static_cast<uint64_t>(concatSources[i].rowMajorIndex);
      concatted[i] = ts[sourceIndex][sourceRowMajorIndex];
    }
    return concatted;
  }

  /** Expand along a single dimension. Example:
   *    values = {0,1,2,3,4,5}
   *    shape  = {2,1,3},
   *    d      = 1,
   *    n      = 4. The output will be:
   *
   *  {0,1,2,0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5,3,4,5}.
   * */
  static std::vector<Number>
  expandSingleDim(const std::vector<Number> &values,
                  const Shape &shape,
                  uint64_t d,
                  uint64_t n) {

    const auto d_i64           = static_cast<int64_t>(d);
    const auto nCopies         = shape.dimProduct(0, d_i64);
    const auto nToCopy         = shape.dimProduct(d_i64, shape.rank_i64());
    const uint64_t nToCopy_u64 = static_cast<uint64_t>(nToCopy);
    std::vector<Number> vOut(shape.nelms_u64() * n);
    uint64_t src = 0;
    uint64_t dst = 0;
    for (int64_t i = 0; i < nCopies; ++i) {
      for (uint64_t j = 0; j < n; ++j) {
        for (uint64_t k = 0; k < nToCopy_u64; ++k) {
          vOut[dst + k] = values[src + k];
        }
        dst += nToCopy;
      }
      src += nToCopy;
    }
    return vOut;
  }

private:
  static std::vector<Number>
  fromIndices(const Data &input, const std::vector<int64_t> &indices) {
    std::vector<Number> out;
    out.reserve(indices.size());
    for (auto i : indices) {
      out.push_back(input.data[static_cast<uint64_t>(i)]);
    }
    return out;
  }
};

// To prevent compiling the ViewChange template class multiple times for the
// same Number type across different translation units, we declare here that
// there already exists a compiled version on another translation unit
// (viewchange.cpp), which will be available to link to.
extern template class ViewChange<int8_t>;
extern template class ViewChange<uint8_t>;
extern template class ViewChange<int16_t>;
extern template class ViewChange<uint16_t>;
extern template class ViewChange<int32_t>;
extern template class ViewChange<uint32_t>;
extern template class ViewChange<int64_t>;
extern template class ViewChange<uint64_t>;
extern template class ViewChange<float>;
extern template class ViewChange<double>;

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
