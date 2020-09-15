// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_SHAPE_HPP
#define POPRITHMS_NDARRAY_SHAPE_HPP

#include <ostream>
#include <vector>

namespace poprithms {

namespace util {
class Permutation;
}

namespace ndarray {

using poprithms::util::Permutation;

/**
 * A class to represent a N-dimensional rectangular volume.
 * */
class Shape {

private:
  std::vector<int64_t> shp;

public:
  using Shapes = std::vector<Shape>;
  using Lower  = decltype(shp);
  using Upper  = decltype(shp);

  Shape(const std::vector<int64_t> &s_) : shp(s_) {}
  Shape(std::vector<int64_t> &&s_) : shp(std::move(s_)) {}

  Shape(const std::initializer_list<int64_t> &s)
      : Shape(std::vector<int64_t>(s)) {}

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

  Shape flatten() const { return Shape({nelms()}); }

  /**
   * \return A Shape which is the same as this, but with all `1's removed.
   *         Note that `0's are not removed.
   * */
  Shape squeeze() const;

  /**
   * \return A copy of this Shape but wth a 1 inserted in dimension \a d. The
   *         returned Shape has rank 1 greater than this Shape's rank.
   * */
  Shape unsqueeze(uint64_t d) const;

  /**
   * \return A copy of this Shape but with \a dim0 prepended.
   *
   * Example: if this is (3,4), then calling prepend(2) returns (2,3,4).
   * */
  Shape prepend(int64_t dim0) const;

  /**
   * Throw an error if the size of \a l or size of \a u is not the same as the
   * rank of this Tensor, or if
   *       l[i] > u[i] or
   *       l[i] < 0 or
   *       u[i] > dim(i),
   * for a dimension "i" less than the rank of this Shape.
   * */
  void assertBoundsAreValid(const Lower &l, const Upper &u) const;

  /**
   * Project the Shape into \a x1 - \a x0 dimensions, by retaining
   * dimensions d in the range
   *  0 <= \a x0 <= d < \a d1 <= rank_u64().
   * */
  Shape dimRange(int64_t x0, int64_t x1) const {
    return {{shp.cbegin() + x0, shp.cbegin() + x1}};
  }

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
   * \return The number of elements in this Shape. It is the product of
   *        dimension sizes.
   * */
  int64_t nelms() const;
  uint64_t nelms_u64() const { return static_cast<uint64_t>(nelms()); }

  int64_t rank_i64() const { return static_cast<int64_t>(rank_u64()); }
  uint64_t rank_u64() const { return shp.size(); }

  int64_t dim(uint64_t d) const { return shp[d]; }
  uint64_t dim_u64(uint64_t d) const { return static_cast<uint64_t>(dim(d)); }

  const std::vector<int64_t> &get() const { return shp; }

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
  Shape broadcast(int64_t N, uint64_t dimension) const;

  /**
   * Reverse the dimensions of this Shape.
   *
   * Example: If this is (2,3,5), then (5,3,2) is returned
   * */
  Shape reverse() const;

  /**
   * Permute the dimensions of this Shape.
   *
   * Example: If this is (2,3,5) and \a p is (1,2,0), then (3,5,2) is
   *returned.
   **/
  Shape dimShuffle(const Permutation &p) const;

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
   * \return The column major indices in this Shape obtained by slicing
   *
   * \sa Shape::getSlicedRowMajorIndices
   * */
  std::vector<int64_t> getSlicedColMajorIndices(const Lower &l,
                                                const Upper &u) const;

  /**
   * \return The row major indices in the Shape resulting from applying
   *         Permutation \a p to this Shape.
   *
   * Example. If this is (2,3), and p is (1,0) -- this corresponds to a simple
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

  void append(std::ostream &os) const;

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

  void assertFlatPoint(int64_t flatPoint) const;

  void assertValidDimension(uint64_t d) const;

  bool operator==(const Shape &rhs) const { return shp == rhs.shp; }
  bool operator!=(const Shape &rhs) const { return shp != rhs.shp; }
};

std::ostream &operator<<(std::ostream &, const Shape &);

} // namespace ndarray
} // namespace poprithms

#endif
