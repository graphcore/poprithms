// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_SHAPE_HPP
#define POPRITHMS_UTIL_SHAPE_HPP

#include <ostream>
#include <vector>

namespace poprithms {
namespace util {

class Shape {

private:
  std::vector<int64_t> shp;

public:
  Shape(const std::vector<int64_t> &s_) : shp(s_) {}
  Shape(const std::initializer_list<int64_t> &s)
      : Shape(std::vector<int64_t>(s)) {}

  template <class T> static Shape fromPartials(const T &t) {
    std::vector<int64_t> shp_(t.size());
    for (uint64_t d = 0; d < t.size(); ++d) {
      shp_[d] = static_cast<int64_t>(t[d].size());
    }
    return Shape(shp_);
  }

  /**
   * The number of elements in this Shape. It is the product of the dimension
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

  bool operator==(const Shape &rhs) const { return shp == rhs.shp; }
  bool operator!=(const Shape &rhs) const { return shp != rhs.shp; }
};

std::ostream &operator<<(std::ostream &, const Shape &);

} // namespace util
} // namespace poprithms

#endif
