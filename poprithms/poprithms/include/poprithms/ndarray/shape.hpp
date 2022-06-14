// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_SHAPE_HPP
#define POPRITHMS_NDARRAY_SHAPE_HPP

#include <array>
#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/ndarray/accessors.hpp>

namespace poprithms {

namespace util {
class Permutation;
}

namespace ndarray {

// Non-intrusive boost based serialization class for serializing classes in
// the ndarray namespace.
class BoostSerializer;

class Shape;
using Shapes = std::vector<Shape>;

using poprithms::util::Permutation;

/**
 * A class to represent a N-dimensional rectangular volume of elements.
 * */

class Shape {

private:
  std::vector<int64_t> shp;
  friend class BoostSerializer;

public:
  using Shapes = std::vector<Shape>;
  using Lower  = decltype(shp);
  using Upper  = decltype(shp);

  Shape(const std::vector<int64_t> &s_) : shp(s_) {}
  Shape(std::vector<int64_t> &&s_) : shp(std::move(s_)) {}

  Shape(const Shape &) = default;
  Shape(Shape &&)      = default;

  Shape &operator=(const Shape &) = default;
  Shape &operator=(Shape &&) = default;

  Shape(const std::initializer_list<int64_t> &s)
      : Shape(std::vector<int64_t>(s)) {}

  template <typename Container> static Shape createFrom(Container &&c) {
    return Shape(std::vector<int64_t>(c.cbegin(), c.cend()));
  }

  /**
   * \param inShapes The Shapes to concatenate.
   * \param axis The dimension to concatenate in.
   *
   * Shapes in \a inShapes must be the same rank and can only differ in
   * dimension \a axis.
   *
   * \return The concatenation of \a inShapes along dimension \a axis.
   *
   * Example: inShapes=((2,3),(2,4)) and axis=1
   *          return (2,7).
   * */
  static Shape concat(const Shapes &inShapes, uint64_t axis);

  /**
   * \return The indices in concatenation dimension \a axis where the input
   *         Shapes \a inShapes touch. The returned vector of indices is of
   *         size 1 greater than the rank of the input Shapes. It is the
   *         cumulative sum of the sizes of \a inShapes along dimension \a
   *         axis.
   *
   * Example: inShapes=((2,1),(2,2),(2,3)) and axis=1
   *          return (0,1,3,6).
   * */
  static std::vector<int64_t> concatPartitionPoints(const Shapes &inShapes,
                                                    uint64_t axis);

  /**
   * Equivalent to Shape::concat({*this, rhs}, axis).
   * */
  Shape concat(const Shape &, uint64_t axis) const;

  /**
   * \return true iff \a rhs has the same rank as this Shape, and if \a rhs
   *         and this Shape have the same sizes in every dimension which is
   *         not \a axis.
   * */
  bool concattable(const Shape &rhs, uint64_t axis) const;

  /**
   * Throws an error if concattable(rhs, axis) is false.
   * */
  void assertConcattable(const Shape &rhs, uint64_t axis) const;

  /**
   * Assert that a tensor of this Shape can be one-hot encoded at positions
   * defined by a tensor of shape #indices. More specifically, this Shape must
   * be rank-2 for example (N, C) and \a indices must be rank-1, (N,). If
   * these conditions are not satisfied, a desciptive error is thrown.
   *
   * Examples of valid and invalid Shapes:
   * this     indices
   * ----     -------
   * (5,3)    (5)      valid
   * (5,3)    (3)      invalid
   * (5,4,3)  (5)      invalid
   * (5)      (5)      invalid
   * (5,3)    ()       invalid
   * */
  void assertOneHotEncodeable(const Shape &indices) const;

  /**
   * Throws an error if either
   * 1) shapes is empty, or
   * 2) any 2 Shapes cannot be concatenated.
   * */
  static void assertConcattable(const Shapes &shapes, uint64_t axis);

  /**
   * Suppose this Shape is (3,4), representing a tensor
   *
   *   xxxx
   *   xxxx
   *   xxxx
   *
   * This tensor is to be updated by another tensor, the "updater" of Shape
   * (3,2),
   *
   *    uu
   *    uu
   *    uu
   *
   * The updater replaces a region of the tensor to be updated, resulting in,
   * for example
   *
   *   xuux      xxuu       uuxx
   *   xuux  or  xxuu   or  uuxx
   *   xuux      xxuu       uuxx
   *
   * The dimensions \a dims represent the free dimensions at which to apply
   * the update, and some "offset" defines what the indices in the free
   * dimensions are. "offset" is of shape \a offsetShape.
   *
   * Restrictions are:
   *
   * 1) updated (this) tensor and updater tensor have same rank.
   * 2) #dims are sorted, unique, and all less than the rank of this tensor
   * 3) offsetShape is rank-1 and of the same size as dims
   * 4) in all dimensions not in #dims, updater and updated have the same size
   *
   * This method asserts that 1-4 are true, throwing an error if not.
   * */
  void assertDynamicUpdate(const Shape &updater,
                           const Dimensions &dims,
                           const Shape &offsetShape) const;

  Shape flatten() const { return Shape({nelms()}); }

  /**
   * Replace this Shape's dimensions in the range [from, to) with #newDims.
   *
   * Suppose this Shape is (1,4,1,5,6). Here are some example inputs/outputs:
   *
   * from   to   newDims     result
   * ----   ---  -------     ------
   *  0     0    {1}         (1,1,4,1,5,6)
   *  0     3    {2,2}       (2,2,5,6)
   *  0     5    {24,5}      (24,5)
   *  2     3    {}          (1,4,5,6)
   *
   * The resulting Shape must have the same number of elements, otherwise an
   * error is thrown. #from and #to must satisy from <= to <= rank.
   * */
  Shape reshapePartial(uint64_t from,
                       uint64_t to,
                       const std::vector<int64_t> &newDims) const;

  /**
   * Reshape to a rank-2 Shape, where the size of the first dimension is the
   * product of dimensions in [0, axis). Specifically, if axis = 0, the
   * returned Shape is (1,nelms) and if axis = rank, the returned Shape is
   * (nelms, 1).
   * */
  Shape flattenTo2d(uint64_t axis) const;

  /**
   * Reshape by collapsing all dimensions in [from, to) into a single
   * dimensions. For example, if this Shape is (2,3,5,7). then flatten(1,3) is
   * (2,15,7).
   *
   * \param from The first dimension in the collapse sequence.
   * \param to One past the final dimension to collapse.
   *
   * It is required that 0 <= from < to <= rank(). */
  Shape flatten(uint64_t from, uint64_t to) const &;
  Shape &&flatten(uint64_t from, uint64_t to) &&;

  /**
   * \return A Shape which is the same as this, but with all `1's removed.
   *         Note that `0's are not removed.
   * */
  Shape squeeze() const;

  /**
   * Squeeze out 1's in certain dimensions.
   *
   * \return A Shape which is the same as this, but with `1's in dimensions \a
   *         dims removed. If the dimension size of any dimension in \a dims
   *         is not 1, an error is thrown.
   *
   * Example:
   *   If this is (1,4,1,5,1) and dims is (0,4), then (4,1,5) is returned.
   * */
  Shape squeeze(const std::vector<uint64_t> &dims) const;

  /**
   * \return A copy of this Shape but wth a 1 inserted in dimension \a d. The
   *         returned Shape has rank 1 greater than this Shape's rank.
   *
   * \param d the dimension at which to insert a 1. d must be be less than or
   *          equal to the rank of this Shape.
   *
   * Example: If this is (3,4) and d is 0, then (1,3,4) is returned.
   *
   * Example: If this is (3,4) and d is 2, then (3,4,1) is returned.
   * */
  Shape unsqueeze(uint64_t d) const;

  /**
   * A copy of this Shape but with 1's inserted in specific dimension.
   *
   * \param dims The dimensions where the output Shape will have 1's inserted.
   *             \a dims must not contain duplicates.
   *
   * Example: If this is (3,4) and dims=(0,2,3) then (1,2,1,1,3,4) is returned
   *
   * Example: If this is (3,4) and dims=(2,3) then (3,4,1,1) is returned.
   *
   * */
  Shape unsqueeze(const std::vector<uint64_t> &dims) const;

  /**
   * Insert 1's into this Shape. Unlike #unsqueeze, the dimensions #dims are
   * the dimensions in the Shape before it is reshaped. This method
   * corresponds to poplar's expand.
   *
   * Examples:
   *  Shape({2}).expand({0}) == Shape({1,2})
   *  Shape({2}).expand({1}) == Shape({2,1})
   *  Shape({3,4}).expand({0,1,1}) == Shape({1,3,1,1,4});
   *
   * Duplicates are allowed, and all dimensions must be in the range [0,
   * rank].
   * */
  Shape insertOnesAt(const std::vector<uint64_t> &dims) const;

  /**
   * \return A copy of this Shape but with \a dim0 prepended.
   *
   * Example: if this is (3,4), then calling prepend(2) returns (2,3,4).
   * */
  Shape prepend(int64_t dim0) const;

  /**
   * \return A copy of this Shape but with the dimensions in \a dims0
   *         prepended.
   *
   * Example: If this is (3,4) and #dims0 is (5,6), the calling prepend(dims)
   *          on this Shape returns the Shape (5,6,3,4).
   * */
  Shape prepend(const Shape &dims0) const;

  /**
   * \return A copy of this Shape but with nOnes 1's prepended.
   * */
  Shape prependOnes(uint64_t nOnes) const;

  /**
   * Increase this Shape's rank by 1, by adding the dimension \a dimEnd to the
   * end. For example, this Shape is (2,3) and \a dimEnd is 5, then the
   * returned Shape is (2,3,5).
   * */
  Shape append(int64_t dimEnd) const &;
  Shape &&append(int64_t dimEnd) &&;

  /**
   * Throw an error if the size of \a l or the size of \a u is not the same as
   * the rank of this Shape, or if l[i] > u[i] or l[i] < 0 or u[i] > dim(i),
   * for a dimension "i" less than the rank of this Shape.
   * */
  void assertSliceBoundsAreValid(const Lower &l, const Upper &u) const;
  void assertSliceBoundsAreValid(Dimension d, uint64_t l, uint64_t u) const;

  /**
   * Throw an error if from >= to or if to >= rank.
   * */
  void assertValidFlatten(uint64_t from, uint64_t to) const;

  /**
   * Project the Shape into \a x1 - \a x0 dimensions, by retaining
   * dimensions d in the range
   *  0 <= \a x0 <= d < \a d1 <= rank_u64().
   * */
  Shape dimRange(int64_t x0, int64_t x1) const {
    return {{shp.cbegin() + x0, shp.cbegin() + x1}};
  }

  Shape fromDim(int64_t x0) const { return dimRange(x0, rank_i64()); }

  Shape untilDim(int64_t x1) const { return dimRange(0, x1); }

  /**
   * \return The product of the dimensions in range [x0, x1)
   * */
  int64_t dimProduct(int64_t x0, int64_t x1) const;
  uint64_t dimProduct_u64(int64_t x0, int64_t x1) const;

  /**
   * \return the Shape \a u - \a l.
   * */
  Shape slice(const Lower &l, const Upper &u) const;

  /**
   * Slice this shape in a single dimension. The returned Shape is the same
   * rank as this Shape, and the same size in all dimensions other that \a d,
   * which is of size u - l. */
  Shape slice(Dimension, uint64_t l, uint64_t u) const &;
  Shape &&slice(Dimension, uint64_t l, uint64_t u) &&;

  /**
   * Convert a slice in one dimension into a slice in all dimensions. Example.
   * If this is (2,3,4) and \a d = 1 and \a l = 1 and \a u = 2, then
   * ((0,1,0),(2,2,4)) is returned.
   * */
  std::pair<Lower, Upper>
  getFullSliceBounds(Dimension d, uint64_t l, uint64_t u) const;

  std::pair<Lower, Upper>
  getFullSliceBounds(const Dimensions &,
                     const std::vector<uint64_t> &l,
                     const std::vector<uint64_t> &u) const;

  /** Slice this Shape only in dimension 0.*/
  Shape slice(uint64_t l, uint64_t u) const {
    return slice(Dimension(0), l, u);
  }

  /**
   * Pad this Shape above by \a u and below by \a l
   *
   * \return A Shape of the same rank as this Shape, and of size in dimension
   *         d of this->dim(d) + l[d] + u[d].
   * */
  Shape pad(const Lower &l, const Upper &u) const;

  /**
   * Pad/trim this Shape.
   *
   * \return A shape which has size in dimension d of dim(d) + delta[d].
   * */
  Shape addToDims(const std::vector<int64_t> &delta) const;

  /**
   * Scale this Shape in a single dimension.
   *
   * \param s the factor by which to scale this Shape in a single dimension.
   * \param d the dimension to scale.
   *
   * \return The scaled Shape.
   * */
  Shape scale(Stride s, Dimension d) const &;
  Shape &&scale(Stride s, Dimension d) &&;

  /**
   * Starting indices of a numpy-slice
   * */

  /**
   * Canonicalized/normalized slice parameters.
   * */
  struct NormalizedSliceParams {
    NormalizedSliceParams(const Starts &,
                          const Ends &,
                          const Steps &,
                          const Dims &,
                          const Shape &);

    /** \return index, where 0 <= index < shape.dim(d) */
    int64_t start(uint64_t d) const { return starts[d]; }

    /** \return index, where -1 <= index < shape.dim(d) + 1 */
    int64_t end(uint64_t d) const { return ends[d]; }

    /** \return non-zero step size in dimension d */
    int64_t step(uint64_t d) const { return steps[d]; }

    uint64_t size() const { return starts.size(); }
    void append(std::ostream &) const;

  private:
    // canonicalized versions of constructor arguments, all vectors of size
    // shape.rank_u64()
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> steps;
  };

  NormalizedSliceParams getNormalizedSliceParams(const Starts &,
                                                 const Ends &,
                                                 const Steps &,
                                                 const Dims &) const;

  /** Verify the correctness of \a n, with respect to this Shape. */
  void validateNormalizedSliceParams(const NormalizedSliceParams &n) const;

  /**
   * Numpy-style [start:stop:step] slicing,
   * https://numpy.org/doc/stable/reference/arrays.indexing.html
   *
   * \sa compute::host::Tensor::slice
   * */
  Shape slice(const NormalizedSliceParams &) const;

  /**
   * Numpy-style [start:stop:step] slicing,
   * https://numpy.org/doc/stable/reference/arrays.indexing.html
   *
   * \sa compute::host::Tensor::slice
   * */
  Shape
  slice(const Starts &, const Ends &, const Steps &, const Dims &) const;

  /**
   * \return The number of elements in this Shape. It is the product of
   *        dimension sizes.
   * */
  int64_t nelms() const;
  uint64_t nelms_u64() const { return static_cast<uint64_t>(nelms()); }

  int64_t rank_i64() const { return static_cast<int64_t>(rank_u64()); }
  uint64_t rank_u64() const { return shp.size(); }

  int64_t dim(uint64_t d) const { return shp[d]; }
  uint64_t dim_u64(uint64_t d) const { return static_cast<uint64_t>(dim(d)); }
  int64_t finalDim() const { return shp.back(); }

  const std::vector<int64_t> &get() const { return shp; }
  std::vector<uint64_t> get_u64() const { return {shp.cbegin(), shp.cend()}; }

  /**
   * \return true if this Shape contains no 1's.
   * */
  bool isSqueezed() const;

  /**
   * \return All dimensions which are of size 1, in ascending order.
   * */
  std::vector<uint64_t> singletonDimensions() const;

  /**
   * \return All dimensions which are not of size 1, in ascending order.
   * */
  std::vector<uint64_t> nonSingletonDimensions() const;

  /**
   * Perform numpy binary broadcasting between this Shape and \a rhs. See
   * https://numpy.org/doc/stable/user/basics.broadcasting.html
   *
   * \return The broadcast Shape.
   *
   * Example :  this = (1,3,1) and rhs = (5,1,2), returns (5,3,2).
   * */
  Shape numpyBinary(const Shape &rhs) const;

  /**
   * Perform Shape reduction using numpy repeated binary broadcasting.
   *
   * Example:
   *  (4,2,1,1) and (1,3,1) and (1,5)  returns (4,2,3,5)
   * */
  static Shape numpyVariadic(const Shapes &shapes);

  /**
   * Shape inference for numpy v1.19 matmul. See
   * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
   *
   * \param arg0 The Shape of the first argument in the matrix multiplication
   * \param arg1 The Shape of the second argument in the matrix multiplication
   * \return The Shape of the output of the matrix multiplication
   *  */
  static Shape matmul(const Shape &arg0, const Shape &arg1);

  Shape matmul(const Shape &arg1) const { return matmul(*this, arg1); }

  enum RoundMode { Floor, Ceil };
  using RoundModes = std::vector<RoundMode>;

  /**
   * 1-D pooling, as described at:
   * https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
   *
   * There is a nice animation desribing the parameters here:
   * https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
   *
   * unrounded output size is:
   * ( data + lowPad + uppPad - 1 - dilation * (kernel - 1) ) / stride
   *
   * To obtain the final output size, the unrounded value is floored, if
   * RoundMode::Floor, and otherwise it is rounded up to the nearest integer.
   *
   * \param window The number of values to pool over. For example, if
   *               window = 3, then every value in the output of pooling is
   *               influenced by 3 positions in the input.
   *
   * \param dilation The spacing between positions in the window which
   *                 influence output values. For example, if window = 3 and
   *                 dilation = 4, then the sampling stencil/mask is
   *                 100010001.
   *
   * \param stride The distance between applications of the pooling stencil.
   *
   * \param RoundMode If Floor, applications of the stencil which overflow the
   *                  input are ignored. If Ceil, additional padding is
   *                  implicitly added.
   * */
  static uint64_t pool1d(uint64_t inputSize,
                         uint64_t window,
                         uint64_t lowPad,
                         uint64_t uppPad,
                         Dilation dilation,
                         Stride stride,
                         RoundMode);

  /**
   * 1-d convolution. The receptive field of a convolution kernel is
   * equivalent to the window size in pooling - \sa pool1d.
   * */
  static uint64_t convolve1d(uint64_t data,
                             uint64_t kernel,
                             uint64_t lowPad,
                             uint64_t uppPad,
                             Dilation dilation,
                             Stride stride) {
    return pool1d(
        data, kernel, lowPad, uppPad, dilation, stride, RoundMode::Floor);
  }

  /**
   * N-d pooling. The input parameters #window, #lowPrePads, #uppPrePads,
   * #dilations, and #strides must all be either
   * 1) the same size as this Shape, or
   * 2) of size 0. When parameters are of size 0, they take default values, as
   *    described below.
   *
   * \param window The size of the receptive field, in number of
   *               values/pixels, over which pooling occurs. Default: the
   *               singleton Shape, containing all 1s.
   *
   * \param lowPrePads How much padding to apply below this Shape before
   *                   pooling. Default: all 0.
   *
   * \param uppPrePads How much padding to apply above this Shape before
   *                   pooling. Default: all 0.
   *
   * \param dilations Default: all 1, which corresponds to a contiguous
   *                  receptive window.
   *
   * \param stride Default: all 1, which corresponds to dense sampling.
   *
   * \sa pool1d
   * */
  Shape pool(const Shape &window,
             const std::vector<uint64_t> &lowPrePads,
             const std::vector<uint64_t> &uppPrePads,
             const Dilations &dilations,
             const Strides &strides,
             const RoundMode) const;

  /** The spatial component of a convolution.
   * \sa batchedMultiChannelConvolve
   * */
  Shape convolve(const Shape &kernel,
                 const std::vector<uint64_t> &lowPrePads,
                 const std::vector<uint64_t> &uppPrePads,
                 const Dilations &dilations,
                 const Strides &strides) const {
    return pool(
        kernel, lowPrePads, uppPrePads, dilations, strides, RoundMode::Floor);
  }

  /**
   * Convolution of a batch of multi-channel spatial data, with a
   * multi-channel kernel. This is equivalent to PyTorch's 2-d convolution
   * https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
   * when this Shape has rank 4.
   * Specifically, if this Shape is (N, C, *spatialData)
   * and kernel is (K, *spatialKernel),
   * the returned Shape is (N, C, spatialData.convolve(spatialKernel)).
   *
   * If this Shape has rank 2, then the parameters #lowPrePads, #uppPrePads,
   * #dilations, and #strides must have rank R - 2, or 0.
   * */
  Shape batchedMultiChannelConvolve(const Shape &kernel,
                                    const std::vector<uint64_t> &lowPrePads,
                                    const std::vector<uint64_t> &uppPrePads,
                                    const Dilations &dilations,
                                    const Strides &strides) const;

  static Shape singleton(uint64_t rnk) {
    return Shape(std::vector<int64_t>(rnk, 1));
  }

  /**
   * \param to The Shape to be expanded to. \a to cannot be smaller than this
   *           in any dimension.
   *
   * \return  The indices of this Shape which will be broadcast when it is
   * numpy broadcast with \a to.
   *
   * Example 1:
   *     this [         3      1      5      ]
   *     to   [   2     3      4      5      ]
   *  return  [         false  true   false  ]
   *
   * Example 2:
   *    this  [         1      5      1      1      ]
   *    to    [   2     3      5      7      1      ]
   *  return  [         true   false  true   false  ]
   *
   *  */
  std::vector<bool> numpyWhereToExpand(const Shape &to) const;

  /**
   * Like #numpyWhereToExpand, but returns the indices which evaluate to true
   * instead of boolean mask.
   * */
  std::vector<uint64_t> numpyIndicesToExpand(const Shape &to) const;

  /**
   * The partial distances along axes if the Shape is iterated through in row
   * major order. Recall that row major order means iterating faster along
   * later axes.
   *
   * Example:
   *   this = (2,3,4), returns (12, 4, 1).
   * */
  std::vector<int64_t> getRowMajorStrides() const;

  /**
   * The partial distances along axes if the Shape is iterated through in
   * column major order - faster along ** earlier ** axes.
   *
   * Example:
   *   this = (2,3,4), returns (1, 2, 6).
   * */
  std::vector<int64_t> getColMajorStrides() const;

  /**
   * \return The absolute distance from the zeroth element to \a point, if
   * this Shape is iterated through faster along further-right axes. This is
   *         inner product of \a point with the row major strides.
   * */
  int64_t getRowMajorIndex(const std::vector<int64_t> &point) const;

  /**
   * \return The absolute distance from the zeroth element to \a point if this
   *         Shape is iterated through faster along further-left axes. This is
   *         inner product of \a point with the column major strides.
   * */
  int64_t getColMajorIndex(const std::vector<int64_t> &point) const;

  /**
   * \return The point which has row major index equal to \a index.
   * */
  std::vector<int64_t> getRowMajorPoint(int64_t index) const;

  /**
   * \return The point which has column major index equal to \a index.
   * */
  std::vector<int64_t> getColMajorPoint(int64_t index) const;

  /**
   * \return A copy of this Shape, but with the size of dimension \a dimension
   *         larger by a factor \a N. The retured Shape has the same rank as
   *         this Shape.
   * */
  Shape broadcast(int64_t N, uint64_t dimension) const &;
  Shape &&broadcast(int64_t N, uint64_t dimensions) &&;

  /** \return A copy of this Shape, but with the dimension \a dimension
   *          resized to be N.
   */
  Shape resizeSingleDim(int64_t N, uint64_t dimension) const;

  /**
   * Reverse the dimensions of this Shape.
   *
   * Example: If this is (2,3,5), then (5,3,2) is returned
   * */
  Shape reverse() const;

  /**
   * Permute the dimensions of this Shape.
   *
   * Example: If this is (2,3,5) and \a p is (1 2 0), then (3,5,2) is
   *          returned.
   **/
  Shape dimShuffle(const Permutation &p) const;

  /**
   * Roll dimension \a dimIdx to the dimension \a newIdx
   *
   * The other dimensions remain in the same relative order
   *
   *  \param dimIdx     The dimension to move from
   *  \param newIdx     The dimension to move to
   *  \returns          The Shape of the dimshuffled Tensor
   *
   * Example. If this Shape is (2,3,5,7,11) then dimRoll with
   *          \a dimIdx = 1 and \a newIdx = 3 results in Shape (2,5,7,3,11).
   *
   */
  Shape dimRoll(Dimension dimIdx, Dimension newIdx) const;

  /**
   * Subsample elements from this Shape.
   *
   * \param strides The interval in each dimension between the elements to
   *                sample. Values in \a strides must be strictly positive.
   *
   * Example: if this is (12,5) and strides=(6,2), then the returned Shape is
   *          (ceil(12/6)=2, ceil(5/2)=3)
   *
   * Subsampling starts at element 0 in each dimension.
   * */
  Shape subSample(const std::vector<uint64_t> &strides) const;

  /**
   * \return The row major indices for all points in the outer product of
   *         subPartials.
   *
   * Example :
   * If this is (2,3,5) and subPartials is ((1),(1,2),(0)), return (15, 20).
   * This is because 15 is the row major index of (1,1,0) and 20 is the row
   * major index of (1,2,0).
   * */
  std::vector<int64_t> getRowMajorIndices(
      const std::vector<std::vector<int64_t>> &subPartials) const;

  /**
   * \return The row major indices of the slice \a u - \a l of this Shape.
   *
   * Example: If this=(3,3), \a l is (1,1) and \a u is (3,2), then {4,7} is
   * returned:
   *
   *      0    1    2
   *         +---+
   *      3  | 4 |  5
   *         |   |
   *      6  | 7 |  8
   *         +---+
   *
   * */
  std::vector<int64_t> getSlicedRowMajorIndices(const Lower &l,
                                                const Upper &u) const;

  /**
   * \return The row major indices of performing a numpy-style slice of this
   *         Shape. The numpy-style slice is a combination of poplar-style
   *         slice, dimShuffle, and reverse.
   *         See https://numpy.org/doc/stable/reference/arrays.indexing.html
   *  */
  std::vector<int64_t> getSlicedRowMajorIndices(const Starts &,
                                                const Ends &,
                                                const Steps &,
                                                const Dims &) const;

  std::vector<int64_t>
  getSlicedRowMajorIndices(const NormalizedSliceParams &n) const;

  /**
   * Suppose a reduction operation reduces a Tensor from this Shape to a
   * Tensor of Shape \a outShape. Such a reduction defines a surjection
   * (many-to-one) mapping from this Shape, to \a outShape.
   *
   * This class method returns indices in \a outShape, in row major order,
   * which the elements in this Shape map to.
   *
   * Example: Suppose this Shape is (2,3) and outShape is (2,1). Then the
   * returned vector is (0,0,0,1,1,1). In row major order, elements (0,1,2) in
   * this Shape map to element 0 in outShape, and elements (3,4,5) map to
   * element 1:
   *
   *   0 1 2  --> 0
   *   3 4 5  --> 1
   *
   * */
  std::vector<int64_t> getReducedRowMajorIndices(const Shape &outShape) const;

  /**
   * \return The column major indices in this Shape obtained by slicing
   *
   * \sa Shape::getSlicedRowMajorIndices
   * */
  std::vector<int64_t> getSlicedColMajorIndices(const Lower &l,
                                                const Upper &u) const;

  /**
   * Gather slices of the row major indices of this Shape
   *
   * The slices are of width 1 on axis \a dimension, at positions \a where.
   *
   * The indices returned correspond to the Shape which is identical to this
   * Shape, except in axis \a dimension where it has a size equal to the
   * number of indices in \a where. As an example, if this Shape is (3,5),
   * dimension is 1, and where is {2,2,1,4}, the returned indices is of size
   * equal to 3 * 4 = 12.
   *
   * As another example, if this=(3,3), axis = 1, where = {0,2}, then the
   * returned indices are (0,2,3,5,6,8), corresponding to the slices:
   *
   *   +-----+   +-----+
   *   |  0  | 1 |  2  |
   *   |     |   |     |
   *   |  3  | 4 |  5  |
   *   |     |   |     |
   *   |  6  | 7 |  8  |
   *   +-----+   +-----+
   *
   * The indices in \a where may be duplicated and do not need to be ordered.
   * */
  std::vector<int64_t>
  gatherRowMajorIndices(uint64_t dimension,
                        const std::vector<int64_t> &where) const;

  /** Gather row major indices along all dimensions. See also the 1-d
   * gatherRowMajorIndices.
   *
   * \param where The positions to gather in each dimension. This vector must
   *              have size equal to this Shape's rank.
   * */
  std::vector<int64_t>
  gatherRowMajorIndices(const std::vector<std::vector<int64_t>> &where) const;

  /**
   * Gather slices of the column major indices of this Shape
   *
   * \sa Shape::gatherColMajorIndices
   * */
  std::vector<int64_t>
  gatherColMajorIndices(uint64_t dimension,
                        const std::vector<int64_t> &where) const;

  /**
   * \return The row major indices in the Shape resulting from applying
   *         Permutation \a p to this Shape.
   *
   * Example. If this is (2,3), and p is (1 0) -- this corresponds to a simple
   * 2-D transpose -- then the returned vector is {0,3,1,4,2,5}.
   *
   *  [[0 1 2]        [[0 3]
   *   [3 4 5]]  ->    [1 4]
   *                   [2 5]].
   *
   * */
  std::vector<int64_t>
  getDimShuffledRowMajorIndices(const Permutation &p) const;

  /**
   * \return The row major indices of a Shape which is this Shape, reversed
   *         along the dimensions \a dims.
   *
   * Example. If this is (2,3) and dims=(1), then the returned vector is
   *          {3,4,5,0,1,2}.
   *
   *  [[0 1 2]     [[3 4 5]
   *   [3 4 5]] ->  [0 1 2]]
   * */
  std::vector<int64_t>
  getReversedRowMajorIndices(const std::vector<uint64_t> &dims) const;

  /**
   * \return The row major indices of a Shape which is a subsampling of (the
   *         elements in) this Shape, with independent strides in each
   *        dimension.
   *
   * Example. If this is (2,3) and strides=(1,2), then the returned vector is
   *          {0,2,3,5}.
   *
   *  [[0 1 2]     [[0 2]
   *   [3 4 5]] ->  [3 5]]
   *   */

  std::vector<int64_t>
  getSubSampledRowMajorIndices(const std::vector<uint64_t> &strides) const;

  /**
   * Return the Shapes of which are required to sequentially wrap
   * this Tensor in padding, starting from from dimension 0.
   *
   * Example. If this Shape is (1,2), lower=(0,1),upper=(2,3), then the
   *          return Shapes are (((0,2),(2,2)),((3,1),(3,3))). These are
   *          illustrated below, where the 'xx' denotes this Shape.
   *
   *                100111
   *                100111
   *                1xx111.
   *
   * Example. If this Shape is (10,20), lower=(1,2), upper=(3,4), then the
   *          returned Shapes are (((1,20),(3,20)),((14,2),(14,4)))
   *
   * */
  std::vector<std::array<Shape, 2>>
  getPadShapes(const std::vector<uint64_t> &lower,
               const std::vector<uint64_t> &upper) const;

  /**
   * \return The row major indices in Shape resulting from expanding this
   *         Shape to \a to.
   *
   * Example 1: this=(3,1) and to=(3,2)
   *   Returns: 0,0,1,1,2,2
   *
   * Example 2: this=(1,3) and to=(2,3)
   *   Returns: 0,1,2,0,1,2
   *
   * Example 3: this=(2,1,3) and to=(2,4,3)
   *   Returns:  0,1,2,0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5,3,4,5.
   *   That is:  0,1,2 repeated 4 times then 3,4,5 repeated 4 times.
   *
   * Example 4: this=(2) and to=(10,2)
   *   Returns: 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
   * */
  std::vector<int64_t> getExpandedRowMajorIndices(const Shape &to) const;

  /** A generalization of \a getRowMajorExpandedIndices and
   * \a getRowMajorDimShuffledIndices, where \a strides can be any values.
   *
   * Example 1: this=(2,3) and strides=(3,1)
   *   Returns: 0,1,2,3,4,5
   *
   * Example 2: this=(3,2) and strides=(1,2) (a dimension shuffling example)
   *   Returns: 0,2,4,1,3,5
   *
   * Example 3: this=(3,2) and strides=(0,1) (an expansion example)
   *   Returns: 0,1,0,1,0,1
   *
   * Example 4: this=(3,2) and strides=(4,4)
   *   Returns: 0,4,4,8,8,12
   * */
  std::vector<int64_t>
  getCustomStridedRowMajorIndices(const std::vector<int64_t> &strides) const;

  /**
   * Map the indices in the output Shape to indices in input Shapes.
   *
   * \param shapes The Shapes to concatenate
   * \param axis The axis along which to concatenate the Shapes
   *
   * \return The sources for each row major index of the output Shape.
   *         Specifically, if the returned vector is \p pairs, the source of
   *         the row major index \a i in the concatenated Shape, is pairs[i].
   *
   * Example:
   *  shapes = {(2,2),(2,1)} and axis = 1.
   *
   *    [[. .]    and  [[x]    ->  [[. . x]
   *     [. .]]         [x]]        [. . x]]
   *
   *  Returns {(shapeIndex=0,rowMajorIndex=0) (0,1) (1,0) (0,2) (0,3) (1,1)}.
   * */
  struct ConcatSource {
    uint64_t sourceShapeIndex;
    int64_t rowMajorIndex;
  };
  static std::vector<ConcatSource>
  getRowMajorConcatSources(const Shapes &shapes, uint64_t axis);

  /**
   * Enumerate all indices, ordered by row major blocks. Blocks themselves are
   * row major ordered too. Specifically, this Shape is tiled with \a
   * blockShape regions, which are enumerated in row major order. Within each
   * tile, the ordering is row major. See example below.
   *
   * \param blockShape The Shape of the nested block.
   *
   * Example: If this Shape is (5,5), and blockShape is (2,3):
   *
   * +----------+--------+
   * | 0  1  2  |  3  4  |
   * | 5  6  7  |  8  9  |
   * +----------+--------+
   * | 10 11 12 |  13 14 |
   * | 15 16 17 |  18 19 |
   * +----------+--------+
   * | 20 21 22 |  23 24 |
   * +----------+--------+
   *
   * Then the returned order is:
   * 0 1 2 5 6 7 3 4 8 9 10 11 12 15 16 17 18 19 20 21 22 23 24
   * ----------- ======= -----------------
   *  block 0    block 1      block 2       etc.
   *
   * Enumerating indices in this tiled fashion can be useful for applications
   * to reduce (CPU) cache misses.
   * */
  std::vector<int64_t> getRowMajorBlockOrdered(const Shape &blockShape) const;

  std::vector<int64_t> getRowMajorBlockOrdered(int64_t blockLength) const {
    return getRowMajorBlockOrdered(
        {std::vector<int64_t>(rank_u64(), blockLength)});
  }

  /**
   * \return all the dimensions which this Shape must be reduced along to get
   *        the shape #to. This includes any non-singleton initial dimensions
   *        which this shape has.
   *
   * Example:
   *
   *  this        to       return
   *  ----        --       ------
   *  (1,4,2,5)   (2,1)    (1,3)
   *     =   =
   * */
  Dimensions reductionDimensions(const Shape &to) const;

  void append(std::ostream &os) const;

  /**
   * \return A string representation of this Shape.
   * */
  std::string str() const;

  template <typename Container>
  static Container numpyBinary(const Container &a, const Container &b) {
    bool aIsLonger      = a.size() > b.size();
    auto out            = aIsLonger ? a : b;
    const auto &shorter = aIsLonger ? b : a;
    const auto delta    = out.size() - shorter.size();
    for (auto i = delta; i < out.size(); ++i) {
      out[i] = std::max(out[i], shorter[i - delta]);
    }
    return out;
  }

  static void assertNumpyBroadcastable(const std::vector<int64_t> &a,
                                       const std::vector<int64_t> &b);

  /**
   * Return true if this Shape numpy dominates #b.
   * */
  bool numpyDominates(const Shape &b) const {
    return numpyBinary(b) == *this;
  }

  void assertNumpyDominates(const Shape &b) const;

  void assertFlatPoint(int64_t flatPoint) const;

  void assertValidDimension(uint64_t d) const;

  void assertValidDimensions(const std::vector<uint64_t> &) const;

  void assertSameNumberOfElements(const Shape &) const;

  void validateGatherIndices(uint64_t d,
                             const std::vector<int64_t> &where) const;

  void assertCanExpandTo(const Shape &to) const;

  /**
   * This Shape can be reduced to \a outShape if and only if (iff) this Shape
   * and \a outShape can be added using numpy broadcasting rules:
   * https://numpy.org/doc/stable/user/basics.broadcasting.html
   * */
  bool canReduceTo(const Shape &outShape) const;

  /**
   * If this Shape cannot be reduced to the Shape \a outShape, throw an error.
   * */
  void assertCanReduceTo(const Shape &outShape) const;

  /**
   * \param indices A vector of dimensions in the range [0, rank()), which may
   *                contain repeats.
   *
   * \return A sorted subset of indices, where all values in indices which
   *         appear an even number of times do not appear. All dimensions in
   *         #indices which are singletons are removed, as reversing a
   *         singleton dimension has no effect on a tensor.
   * */
  std::vector<uint64_t>
  getCanonicalReverseIndices(const std::vector<uint64_t> &indices) const;

  /**
   * If this Shape is reshaped to Shape \a to, how are the dimensions
   * distributed?
   *
   * This method assumes that this and \a to are fully squeezed. That is, they
   * contain no singleton dimensions. It also assumes that neither is empty,
   * that is neither contains zero elements.
   *
   * Example 1.
   * this Shape :  (6, 2, 4), and
   * to   Shape :  (2, 3, 8).
   * The first dimension of size 6 is split into 2 dimensions of size 2 and 3,
   * and the second and third dimensions of sizes 2 and 4 respectively are
   * merged into a dimension of size 8:
   *
   *     6   2  4   this Shape
   *    / \  \ /
   *   2   3  8     to
   *
   *  returns ((0),(0),(1,2)).
   *            |   |    |
   *  output dimension 0 comes from input dimension 0
   *                |    |
   *    output dimension 1 comes from input dimension 0
   *                     |
   *       output dimension 2 comes from input dimensions 1 and 2.
   *
   *
   * Example 2. The dimensions are not split cleanly; some input dimensions
   * map to multiple output dimensions. The dimensions are thus distributed
   * multiple times.
   *
   *   2   3   5   7   this
   *    \  |  /|  /|
   *      10   7   3   to
   *
   * returns ((0,1,2),(2,3),(3)). The binning is based on cumulative
   * products of Shapes:
   *
   *  this prod       to prod
   *  ==========      ========
   *  2            <  10       => 0 goes to 0
   *  2*3          <  10       => 1 goes to 0
   *  2*3*5        <  10*7     => 2 goes to 0 and 1
   *  2*3*5*7      =  10*7*3   => 3 goes to 1 and 2.
   *
   *  In this third example, some dimensions factorize cleanly, and others do
   *  not:
   *
   *   2 3 5     6   5  4   this
   *   | | |    / \ / \ |
   *   2 3 5   4   5    6   to
   *
   *  returns ((0),(1),(2),(3),(3,4),(4,5)).
   *
   * This method assumes that neither this Shape nor \a to contain any 1's.
   */
  std::vector<Dimensions> getReshapeFactorization(const Shape &to) const;

  /**
   * Do the reshape dimensions factorize (as in the first example above), or
   * do they straddle (like the second and third examples above) ?
   * */
  bool isOrthogonalReshape(const Shape &to) const;

  /**
   * Consider a tensor of Shape inShape, to which we apply in series:
   *
   * 1) a reshape, to \a shapePreDimShuffle, and then
   * 2) a dimension shuffle by Permutation \a endPerm,
   *
   * to produce an output Tensor of Shape \a outShape. It is sometimes
   * possible to replace this with
   *
   * 1) a dimension shuffle by Permutation \a beginPerm
   * 2) a reshape to Shape \a outShape.
   *
   * That is, it is sometimes possible to swap the positions of the reshape
   * and the dimShuffle while preserving the final positions of elements in
   * the final tensor.
   *
   * For example, for a tensor of shape (2,5),
   *
   * t0.reshape(2,3,5).dimShuffle(1 2 0),
   *    is equivalent to,
   *
   * t0.dimShuffle(1 0).reshape(3,5,2).
   *
   * An example where it is not possible to reverse the order is,
   * t0.reshape(2,3,5).dimShuffle(2 0 1).
   *
   * This method returns a tuple, whose bool denotes if the switching of order
   * is possible, and whose Permutation is the one required if the dimShuffle
   * is performed before the reshape.
   *
   * */
  std::pair<bool, Permutation>
  moveDimShuffleBeforeReshape(const Shape &shapePreDimShuffle,
                              const Permutation &tailPerm) const;

  /**
   * Is it possible replace
   *     tensor.slice(l0, u0).reshape(s0)
   * with,
   *    tensor.reshape(zs11).slice(l1, u1) ?
   *
   * That is, instead of slicing then reshaping a tensor, can you reshape and
   * then slice the tensor, but end up with the same final result?
   *
   * Suppose the tensor being sliced has this Shape as its shape.
   *
   * \param slicedShape the shape of the sliced tensor,
   * \param finalShape the shape of the reshaped (after both slice and reshape
   *                   operations) tensor.
   *
   * \return a tuple with values at indices
   *        (0) true if it possible to perform the reordering, false otherwise
   *        (1) the shape of the reshaped tensor, before being sliced,
   *        (2) the dimensions of the reshaped tensor to slice.
   *
   * Some examples (where (x,y) denotes a tensor of shape (x,y)).
   *    (5,6) -> slice -> (2,6) -> reshape -> (2,3,2)
   * becomes
   *    (5,6) -> reshape -> (5,3,2) -> slice -> (2,3,2).
   * and returns (true, (5,3,2), (0))
   *
   *    (5,6) -> slice -> (2,6) -> reshape(4,3)
   * is not not possible to change (returns (false, {}, {})).
   *
   *    (3) -> slice -> (1) -> reshape -> (1,1)
   * might become
   *    (3) -> reshape(3,1) -> slice -> (1,1)
   * and return (true, (3,1), (0)), although this is not a unique solution and
   * (true, (1,3), (1)) might be returned instead.
   * */

  std::tuple<bool, Shape, Dimensions>
  moveReshapeBeforeSlice(const Shape &slicedShape,
                         const Shape &finalShape) const;

  /**
   * Similar to #moveReshapeBeforeSlice, this method attempts to change
   * reshape->slice to slice->reshape. The implementation logic overlaps with
   * #moveReshapeBeforeSlice, as one reordering is possible if and only if the
   * other is.
   *
   * Returns a tuple where
   * 0) the bool says if the rearrangement (to slice->reshape) is possible.
   *
   * 1) the Shape is that of the slice before the reshape (if possible,
   *    otherwise undefined).
   *
   * 2) the Dimensions are the new slice dimensions (if possible, otherwise
   *    undefined).
   * */
  std::tuple<bool, Shape, Dimensions>
  moveSliceBeforeReshape(const Shape &reshape,
                         const Shape &finalSlicedShape) const;

  /**
   * \return The number of dimensions of this Shape of size s.
   * */
  uint64_t nDimsOfSize(int64_t s) const;

  bool operator==(const Shape &rhs) const { return shp == rhs.shp; }
  bool operator!=(const Shape &rhs) const { return shp != rhs.shp; }
  bool operator<(const Shape &rhs) const { return shp < rhs.shp; }
  bool operator>(const Shape &rhs) const { return shp > rhs.shp; }
  bool operator<=(const Shape &rhs) const { return shp <= rhs.shp; }
  bool operator>=(const Shape &rhs) const { return shp >= rhs.shp; }

  /**
   * Return true if, when the dimensions #revDims of a tensor of this shape
   * are reversed, the row-major order of the elements of the tensor change.
   *
   * There row-major order changes only if some of the dimensions in #revDims
   * correspond to non-singleton dimensions. This method assumes that #revDims
   * are all unique.
   * */
  bool reversePreservesOrder(const Dimensions &revDims) const;

  /**
   * Suppose the permutation #p is applied to a tensor of this shape. This
   * method returns true if the row-major order of the elements in the
   * resulting tensor are different.
   *
   * In the absence of singleton dimensions, this is true if and only if #p is
   * the identity permutation (0 1 2 ... N-1), and singleton dimensions can be
   * ignored.
   * */
  bool dimShufflePreservesOrder(const Permutation &) const;

  /**
   * The dimensions of this shape, [0,...,rank).
   * */
  Dimensions dimensions() const;
};

std::ostream &operator<<(std::ostream &, const Shape &);
std::ostream &operator<<(std::ostream &ost,
                         const Shape::NormalizedSliceParams &);

} // namespace ndarray
} // namespace poprithms

#endif
