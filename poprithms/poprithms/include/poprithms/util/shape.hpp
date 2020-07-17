// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_SHAPE_HPP
#define POPRITHMS_UTIL_SHAPE_HPP

#include <ostream>
#include <vector>

namespace poprithms {
namespace util {

/**
 * A class to represent a N-dimensional rectangular volume.
 * */
class Shape {

private:
  std::vector<int64_t> shp;
  using Lower = decltype(shp);
  using Upper = decltype(shp);

public:
  Shape(const std::vector<int64_t> &s_) : shp(s_) {}
  Shape(const std::initializer_list<int64_t> &s)
      : Shape(std::vector<int64_t>(s)) {}

  /**
   * \param inShapes The Shapes to concatenate.
   *
   * \param axis The dimension to concatenate in.
   *
   * All Shapes in `inShapes' must be the same rank and can only differ in
   * dimension `axis'.
   *
   * \return The concatenation of `inShapes' along dimension `axis'.
   *
   * Example: inShapes=((2,3),(2,4)) and axis=1
   *          return (2,7).
   * */
  static Shape concat(const std::vector<Shape> &inShapes, uint64_t axis);

  /**
   * \return The indices along concatenation dimension `axis' where the input
   *         Shapes `inShapes' touch. The returned vector of indices is of
   *         size inShapes.size() + 1, and it is the cumulative sum of the
   *         sizes of `inShapes' along dimension `axis'.
   *
   * Example: inShapes=((2,1),(2,2),(2,3)) and axis=1
   *          return (0,1,3,6).
   * */
  static std::vector<int64_t>
  concatPartitionPoints(const std::vector<Shape> &inShapes, uint64_t axis);

  /**
   * This is equivalent to Shape::concat({*this, rhs}, axis).
   * */
  Shape concat(const Shape &, uint64_t axis) const;

  /**
   * \return true iff `rhs' has equal rank to this Shape, and `rhs' and this
   *         Shape have the same sizes in every dimension which is not `axis'.
   * */
  bool concattable(const Shape &rhs, uint64_t axis) const;

  /**
   * Throws an error if concattable(rhs, axis) is false.
   * */
  void assertConcattable(const Shape &rhs, uint64_t axis) const;

  Shape flatten() const { return Shape({nelms()}); }

  /**
   * \return A Shape which is the same same as this but with all `1's removed.
   *         Note that `0's are not removed.
   * */
  Shape squeeze() const;

  /**
   * \return A copy of this Shape but wthi a 1 inserted in dimension `d'. The
   *         returned Shape has rank 1 greater than this Shape's.
   * */
  Shape unsqueeze(uint64_t d) const;

  /**
   * Throw an error if
   *   l > u or
   *   l < 0 or
   *   u > shape().
   * */
  void assertBoundsAreValid(const Lower &l, const Upper &u) const;

  /**
   * \return the Shape "u - l" if assertBoundsAreValid(l,u).
   * */
  Shape slice(const Lower &l, const Upper &u) const;

  /**
   * The number of elements in this Shape. It is the product of dimension
   * sizes.
   * */
  int64_t nelms() const;
  uint64_t nelms_u64() const { return static_cast<uint64_t>(nelms()); }

  uint64_t rank_u64() const { return shp.size(); }
  int64_t dim(uint64_t d) const { return shp[d]; }

  const std::vector<int64_t> &get() const { return shp; }

  /**
   * Perform numpy binary broadcasting with rhs.
   *
   * \return The broadcast Shape.
   *
   * Example :  this = (1,3,1) and rhs = (5,1,2), returns (5,3,2).
   * */
  Shape numpyBinary(const Shape &rhs) const;

  /**
   * \param to The Shape to be expanded to. "to" cannot be smaller than this
   *           in any dimension.
   *
   * \return  The indices of this Shape which will be broadcast if numpy
   *           broadcast with "to".
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
   * The partial distances along axes if the Shape is iterated through faster
   * along later axes.
   *
   * Example:
   *   this = (2,3,4), returns (12, 4, 1).
   * */
  std::vector<int64_t> getRowMajorStrides() const;

  /**
   * The partial distances along axes if the Shape is iterated through faster
   * along earlier axes.
   *
   * Example:
   *   this = (2,3,4), returns (1, 2, 6).
   * */
  std::vector<int64_t> getColMajorStrides() const;

  /**
   * \return The absolute distance from the zeroth element to "point" if this
   *         Shape is iterated through faster along later axes. This is
   *         inner product of "point" with the row major strides.
   * */
  int64_t getRowMajorIndex(const std::vector<int64_t> &point) const;

  /**
   * \return The absolute distance from the zeroth element to "point" if this
   *         Shape is iterated through faster along earlier axes. This is
   *         inner product of "point" with the column major strides.
   * */
  int64_t getColMajorIndex(const std::vector<int64_t> &point) const;

  /**
   * \return The point which has row major index equal to "index".
   * */
  std::vector<int64_t> getRowMajorPoint(int64_t index) const;

  /**
   * \return The point which has column major index equal to "index".
   * */
  std::vector<int64_t> getColMajorPoint(int64_t index) const;

  /**
   * \return A copy of this Shape but with the size of dimension "dimension"
   *         larger by a factor "N". The retured Shape has the same rank as
   *         this Shape.
   * */
  Shape broadcast(int64_t N, uint64_t dimension) const;

  //

  /**
   * \return The row major indices for all points in the outer product of
   *         subPartials.
   *
   * Example :
   * If this is (2,3,5) and subPartials is ((1),(1,2),(0)), return (15, 20).
   * */
  std::vector<int64_t> getRowMajorIndices(
      const std::vector<std::vector<int64_t>> &subPartials) const;

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

} // namespace util
} // namespace poprithms

#endif
