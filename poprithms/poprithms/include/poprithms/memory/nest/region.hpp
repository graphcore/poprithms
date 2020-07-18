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
 * A set of elements of a Shape. The set is expressed as the outer product of
 * Setts in each of the dimensions. A Sett (see sett.hpp) is a generalization
 * of an interval, and Regions can represent non-contiguous areas within a
 * Shape.
 *
 * A Region is defined by its 2 class members,
 *
 *  1) Shape shape_;
 * defines the containing rectangular volume.
 *
 *  2) std::vector<Sett> setts_;
 * defines the striping pattern of elements in the volume.
 *
 * Examples of 2-d Regions, using 1 to denote a contained element:
 *
 * Example 1:
 * shape_ = (4,7), setts_ = (((2,2,1)), ((2,5,3))):
 *  .......
 *  ...11..
 *  ...11..
 *  .......
 *
 * As described in sett.hpp, Sett=(2,5,3) is used to represent a repeating
 * pattern of
 *   on  for  2, then
 *   off for  5, with a
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
 * Any set of elements which can be expressed independently in each dimension
 * can be expression. A set of elements such as
 *
 * .1.1.1.
 * 1.1.1.1
 * .1.1.1.
 *
 * cannot be expressed by a Region, but can be by the union of 2 Regions:
 *
 * .1.1.1.       .......
 * .......  and  1.1.1.1
 * .1.1.1.       .......
 *
 * Complex Regions result from sequences of slices, concatentations and
 * reshapes of Shapes.
 * */
class Region {

public:
  /**
   * \param shape The rectangular volume which contains this Region
   *
   * \param setts The elements of the containing volume in this
   *              Region are defined by the outer-product of these Setts.
   * */
  Region(const Shape &shape, const std::vector<Sett> &setts);

  /**
   * Example : shape=(10), lower=(3), upper=(9) is equivalent to
   *           constructing with setts=((on=6, off=4, phase=3)).
   *
   * */
  static Region fromBounds(const Shape &shape,
                           const std::vector<int64_t> &lower,
                           const std::vector<int64_t> &upper);

  /**
   * Construct a Region with always-on Setts in all dimensions, except in
   * dimension "dim" which has a depth-1 Sett defined by "st".
   * */
  static Region fromStripe(const Shape &, uint64_t dim, const Stripe &st);

  Region(const Region &rhs) = default;
  Region(Region &&)         = default;
  Region &operator          =(const Region &);

  /**
   * \return Region which contains all elements of "shape".
   * */
  static Region createFull(const Shape &shape);

  /**
   * \return A Region which contains no elements, contained in volume "shape".
   * */
  static Region createEmpty(const Shape &shape);

  const Shape &shape() const { return shape_; }

  uint64_t rank_u64() const { return shape().rank_u64(); }

  int64_t dim(uint64_t d) const { return shape().dim(d); }

  /**
   * \return The total number of elements in this Region.
   * */
  int64_t totalElms() const;

  /**
   * \return The number of elements defined by the Sett along dimension "dim".
   * */
  int64_t nelms(uint64_t dim) const;

  /**
   * \return The number of elements defined by each Setts. The total number of
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
   * \return The intersection of this Region and rhs. The returned Regions
   *         have the same containing Shape as this Region.
   * */
  DisjointRegions intersect(const Region &rhs) const;

  /**
   * A generalization of slicing and sub-sampling.
   *
   * \param where The Region which defines the indices of this Region to
   *              select. It must have the same containing Shape as this.
   *
   * \return DisjointRegions, whose containing Shape is equal to
   * where.nelms().
   *
   * Example
   *
   *  this       where                 returned Regions
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
   * 1.1     ..11.1.          1. 1      ..1..1.
   * 1.1     .......   =>               .......
   * ...     ..11.1.          1. 1      ..1..1.
   *         ..11.1.          .. .      .......
   *
   * As seen in the example above, all the '1's in "scaffold" are replaced
   * by the values in this Region. scaffold.nelms() must equal this Region's
   * containing Shape, in this example this is (3,3).
   * */
  DisjointRegions settFillInto(const Region &scaffold) const;

  /**
   * The reverse of settFillInto, an example is:
   *
   * ink       this                     return
   *
   * 1.1     ..11.1.          1. 1      ..1..1.
   * 1.1     .......    =>              .......
   * ...     ..11.1.          1. 1      ..1..1.
   *         ..11.1.          .. .      .......
   * */
  DisjointRegions settFillWith(const Region &ink) const;

  /**
   * Slice this Region. The Shape of the returned Region is "upper - lower"
   * */
  Region slice(const std::vector<int64_t> &lower,
               const std::vector<int64_t> &upper) const;

  /**
   * The inverse operation expand. Example:
   *
   *   2 3 4 5  this Region's Shape
   *     1 4 1  the output Shape
   *
   * If this Region is not empty, the returned Region's Setts are always on
   * where the output Shape has a singleton dimension, elsewhere they are
   * unchanged from the input Region's Setts.
   * */
  Region reduce(const Shape &outShape) const;

  /**
   * Reshape this Region.
   *
   * Example: If this Region has
   * shape=(2,8) and setts=((),((5,3,0))), and to=(4,4):
   *
   *                                     returned
   *                                  DisjointRegions:
   *               1111                1111   ....
   * 11111...  =>  1...        =       .... + 1...
   * 11111...      1111                1111   ....
   *               1...                ....   1...
   *
   * \param to Shape with the same number of elements as this Region's Shape.
   * */
  DisjointRegions reshape(const Shape &to) const;

  /**
   * Reshape this Region to rank-1.
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
   * \return A debug string.
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
  uint64_t rank_u64() const { return shape().rank_u64(); }

  /**
   * \param rs a vector of disjoint Regions of the same Shape. If the Regions
   *           are not all disjoint and of the same shape, the behaviour of
   *           the object constructed is undefined.
   * */
  explicit DisjointRegions(const Shape &, const std::vector<Region> &rs);

  explicit DisjointRegions(const Shape &sh,
                           const std::vector<std::vector<Sett>> &se)
      : DisjointRegions(sh, regsFromSetts(sh, se)) {}

  static DisjointRegions createEmpty(const Shape &s) {
    return DisjointRegions(s, std::vector<Region>{});
  }

  static DisjointRegions createFull(const Shape &s) {
    return DisjointRegions(s, {Region::createFull(s)});
  }

  DisjointRegions(const DisjointRegions &) = default;
  DisjointRegions(DisjointRegions &&regs)  = default;
  DisjointRegions &operator                =(const DisjointRegions &);

  bool disjoint(const DisjointRegions &rhs) const;

  DisjointRegions(const Region &s) : sh_(s.shape()), regs_({s}) {}

  size_t size() const { return regs_.size(); }

  bool empty() const { return regs_.empty(); }

  const std::vector<Region> &get() const { return regs_; }

  const Region &at(size_t i) const { return regs_[i]; }

  /**
   * \return true iff The Regions are mutually disjoint and have the same
   *         containing Shape.
   * */
  bool isValid() const;
  void confirmValid() const;

  int64_t totalElms() const;

  //  The following methods are the vector extensions of their corresponding
  //  single Region versions.

  std::vector<int64_t> nelms() const;

  DisjointRegions flatten() const;

  DisjointRegions reduce(const Shape &) const;

  DisjointRegions slice(const std::vector<int64_t> &lower,
                        const std::vector<int64_t> &upper) const;

  DisjointRegions settFillInto(const Region &) const;

  DisjointRegions reverse(const std::vector<uint64_t> &dimensions) const;

  DisjointRegions reshape(const Shape &) const;

  DisjointRegions permute(const Permutation &) const;

  DisjointSetts flattenToSetts() const;
};

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
