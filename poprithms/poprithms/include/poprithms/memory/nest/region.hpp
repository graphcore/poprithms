// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_NEST_REGION_HPP
#define POPRITHMS_MEMORY_NEST_REGION_HPP

#include <sstream>
#include <vector>

#include <poprithms/memory/nest/optionalset.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace util {
class Permutation;
}
} // namespace poprithms

namespace poprithms {
namespace memory {
namespace nest {

using Shape       = poprithms::util::Shape;
using Permutation = poprithms::util::Permutation;

class Region;
using OptionalRegion = poprithms::memory::nest::OptionalSet<1, Region>;

class DisjointRegions;

/**
 * A set of elements of an n-d array (a Tensor). The set is expressed as the
 * outer-product of Setts in each of the d- dimensions. A Sett (see sett.hpp)
 * is a generalization of an interval, and so Regions can represent
 * non-contiguous areas within Tensors. In general, any striped sub-region of
 * a Tensor is representable.
 *
 * A Region is completely defined by 2 class members,
 *
 *   > Shape shape_;
 * defines the n-d array in which the Region is contained.
 *
 *   > std::vector<Sett> setts_;
 * defines which elements of the n-d array are contained in the Region.
 *
 * Some examples of 2-d Regions, using 1 to denote a contained element:
 *
 * Example 1:
 * shape_ = (4,7), setts_ = (((2,2,1)), ((2,5,3))):
 *  .......
 *  ...11..
 *  ...11..
 *  .......
 *
 * as described in sett.hpp, Sett=(2,5,3) is used to represent a repeating
 * pattern of
 *   on for 2, then
 *   off for 5, with a
 *   phase of 3:
 *  ...11.....11.....11.....11.....11.....11.....11
 *
 * Example 2:
 * shape_ = (4,7), setts_ = ((), ((2,5,3))):
 *  ...11..
 *  ...11..
 *  ...11..
 *  ...11..
 *
 * as described in sett.hpp, the stripeless Sett (), is always on.
 *
 * Example 3:
 * shape_ = (4,7), setts_ = ((), ((1,1,0))):
 *  1.1.1.1
 *  1.1.1.1
 *  1.1.1.1
 *  1.1.1.1
 *
 *  Example 4:
 *  shape_ = (5,12), setts_ = (((1,1,1)), ((8,4,3),(1,1,0)))
 *  ............
 *  ...1.1.1.1..
 *  ............
 *  ...1.1.1.1..
 *  ............
 *
 * In general, any sub-region which can be expressed independently in each
 * dimension can be expression. A sub-region such as
 *
 * .1.1.1.
 * 1.1.1.1
 * .1.1.1.
 *
 * cannot be expressed by a Region, but can be decomposed into 2 smaller set
 * which can:
 *
 * .1.1.1.       .......
 * .......  and  1.1.1.1
 * .1.1.1.       .......
 *
 * Complex Regions can result from a sequence of slices, concatentations and
 * reshapes of n-d arrays.
 * */
class Region {

public:
  /**
   * \param shape The Shape of the n-d array which contains this Region
   *
   * \param setts The elements of the containing n-d array which are in this
   *              Region are defined by the outer-product of these Setts.
   *
   * setts and shape must have the same size.
   * */
  Region(const Shape &shape, const std::vector<Sett> &setts);

  /**
   * Create a Region which contains all elements of the n-d array defined by
   * shape
   * */
  static Region createFull(const Shape &shape);

  const Shape &shape() const { return shape_; }

  uint64_t rank_u64() const { return shape().rank_u64(); }

  int64_t dim(uint64_t d) const { return shape().dim(d); }

  /**
   * \return The total number of elements in the Region.
   * */
  int64_t totalElms() const;

  /**
   * \return The number of elements defined by the Sett along dimension "dim".
   * */
  int64_t nelms(uint64_t dim) const;

  /**
   * \return The number of elements defined by all Setts. The total number of
   *         elements in the Region is the product of these values.
   * */
  std::vector<int64_t> nelms() const;

  const std::vector<Sett> &setts() const { return setts_; }

  const Sett &sett(uint64_t d) const { return setts()[d]; }

  /**
   * \return true iff the Region contains no elements.
   * */
  bool empty() const;

  /**
   * \return true iff this Region contains all elements of containg n-d array.
   * */
  bool full() const;

  /**
   * \param rhs A Region with the same containing Shape as this.
   *
   * \return The intersection of this Region and rhs. This returned Region has
   *         the same containing Shape as this.
   * */
  DisjointRegions intersect(const Region &rhs) const;

  /**
   * A generalization of slice and subSample operations.
   *
   * \param where The Region which defines the indices of this Region to
   *              select. "where" must have the same containing Shape as this
   *              Region.
   *
   * \return A Region, whose containing Shape is equal to where.nelms().
   *
   * Example
   *
   *  this       where                 returned Region
   * .......    1.1.1.1      . . . .     ....
   * ..1111.    .......   =>             .11.
   * ..1111.    1.1.1.1      . 1 1 .
   * .......    .......
   * */

  DisjointRegions settSample(const Region &where) const;

  /**
   * Fill/scatter this Region into another Region.
   *
   * \param scaffold The Region which this Region should fill.
   *
   * Example:
   *
   * this    scaffold                   return
   *
   * 1.1     ..11.1.        --1.-1-     ..1..1.
   * 1.1     .......   =>   -------     .......
   * ...     ..11.1.        --1.-1-     ..1..1.
   *         ..11.1.        --..-.-     .......
   *
   * As seen in the example, all the '1's in the scaffold are replaced by the
   * values in this Region. scaffold.nelms() must equal this Region's
   * containing Shape, in this example it is (3,3).
   * */
  DisjointRegions settFillInto(const Region &scaffold) const;

  /**
   * The reverse of settFillInto, an example is:
   *
   * ink      this                     return
   *
   * 1.1     ..11.1.        --1.-1-     ..1..1.
   * 1.1     .......   =>   -------     .......
   * ...     ..11.1.        --1.-1-     ..1..1.
   *         ..11.1.        --..-.-     .......
   * */
  DisjointRegions settFillWith(const Region &ink) const;

  /**
   * Slice this Region. The returned DisjointRegions contains 0 Regions if the
   * slice is empty, and 1 Region otherwise. The Shape of the returned object
   * is the difference between lower and upper.
   * */
  DisjointRegions slice(const std::vector<int64_t> &lower,
                        const std::vector<int64_t> &upper) const;

  /**
   * Reshape this Region.
   *
   * Example: If this has shape_=(2,8) and setts_=((),((5,3,0))), and
   * to=(4,4):
   *
   *                                     returned
   *                                  DisjointRegions:
   *               1111                1111   ....
   * 11111...  =>  1...        =       .... + 1...
   * 11111...      1111                1111   ....
   *               1...                ....   1...
   *
   * */
  DisjointRegions reshape(const Shape &to) const;

  /**
   * Reshape this Region to a rank-1 Tensor
   * */
  Region flatten() const;

  /**
   * Reverse this Region at specified indices. The containing Shape is
   * unchanged.
   * */
  Region reverse(const std::vector<uint64_t> &where) const;

  /**
   * A generalized transpose, where the axes are permuted.
   * */
  Region permute(const Permutation &) const;

  /**
   * Expand this Region. This is equivalent to numpy.broadcast_to
   * */
  Region expand(const Shape &to) const;

  /**
   * Attempt to merge this Region with "other". If not possible, the returned
   * object is empty.
   * */
  OptionalRegion merge(const Region &other) const;

  /**
   * Append debug information.
   * */
  void append(std::ostream &ss) const;

  /**
   * \return A debug information.
   * */
  std::string str() const;

  /**
   * \return true iff rhs and this have an empty intersection.
   * */
  bool disjoint(const Region &rhs) const;

  /**
   * \return true iff rhs and this have exactly the same elements and
   *         containing Shape.
   * */
  bool equivalent(const Region &rhs) const;

  /**
   * \return true iff all elements in rhs are also in this, and rhs has the
   *         same containing Shape as this.
   * */
  bool contains(const Region &rhs) const;

  /**
   * \return true iff a and b contain the same elements and have the same
   *         containing Shapes.
   * */
  static bool equivalent(const DisjointRegions &a, const DisjointRegions &b);

private:
  Shape shape_;
  std::vector<Sett> setts_;

  DisjointRegions unflatten(const Shape &to) const;

  void validateBounds(const std::vector<int64_t> &lower,
                      const std::vector<int64_t> &upper) const;

  void confirmSameShape(const Region &) const;

  void confirmShape(const Shape &) const;
};

std::ostream &operator<<(std::ostream &, const Region &);
std::ostream &operator<<(std::ostream &, const DisjointRegions &);
std::ostream &operator<<(std::ostream &, const std::vector<Region> &);

/**
 * A union of disjoint Regions.
 * */
class DisjointRegions {

private:
  Shape sh_;
  std::vector<Region> regs_;

  static std::vector<Region>
  regsFromSetts(const Shape &, const std::vector<std::vector<Sett>> &);

public:
  const Shape &shape() const { return sh_; }

  /**
   * \param rs a vector of disjoint Regs of the same Shape. If the Regs in rs
   * are not all disjoint and of the same shape, the behavior of the object
   * constructed is undefined.
   * */
  explicit DisjointRegions(const Shape &, const std::vector<Region> &rs);

  explicit DisjointRegions(const Shape &sh,
                           const std::vector<std::vector<Sett>> &se)
      : DisjointRegions(sh, regsFromSetts(sh, se)) {}

  static DisjointRegions createEmpty(const Shape &s) {
    return DisjointRegions(s, std::vector<Region>{});
  }

  DisjointRegions(DisjointRegions &&regs) = default;

  DisjointRegions(const Region &s) : sh_(s.shape()), regs_({s}) {}

  decltype(regs_.begin()) begin() { return regs_.begin(); }

  decltype(regs_.cbegin()) cbegin() const { return regs_.cbegin(); }

  decltype(regs_.end()) end() { return regs_.end(); }

  decltype(regs_.cend()) cend() const { return regs_.cend(); }

  size_t size() const { return regs_.size(); }

  bool empty() const { return regs_.empty(); }

  const std::vector<Region> &get() const { return regs_; }

  std::vector<Region> &get() { return regs_; }

  const Region &operator[](size_t i) const { return regs_[i]; }
  const Region &at(size_t i) const { return regs_[i]; }

  Region &operator[](size_t i) { return regs_[i]; }

  /**
   * \return true iff The Regions are mutually disjoint and have the same
   *         containing Shape.
   * */
  bool isValid() const;
  void confirmValid() const;

  std::vector<int64_t> nelms() const;
  DisjointRegions flatten() const;
  DisjointSetts flattenToSetts() const;
};

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
