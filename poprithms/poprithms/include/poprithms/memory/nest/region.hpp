// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_NEST_REGION_HPP
#define POPRITHMS_MEMORY_NEST_REGION_HPP

#include <sstream>
#include <vector>

#include <poprithms/memory/nest/optionalset.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace nest {

using poprithms::ndarray::Dimension;
using poprithms::ndarray::Dimensions;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::ndarray::Stride;
using poprithms::ndarray::Strides;
using poprithms::util::Permutation;

class Region;
using OptionalRegion = poprithms::memory::nest::OptionalSet<1, Region>;

class DisjointRegions;

/**
 * A set of elements in a Shape. The set is expressed as the outer product of
 * Setts, one Sett for each dimension. A Sett (see sett.hpp) is a
 * generalization of an interval. Regions can represent non-contiguous
 * areas within a Shape.
 *
 * A Region is defined by its 2 class members,
 *
 *  1) Shape shape_;
 *     defines the containing rectangular volume.
 *
 *  2) std::vector<Sett> setts_;
 *     defines the rectilinear striping pattern of elements in the volume.
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
  Region() = delete;

  Region(const Region &rhs) = default;
  Region(Region &&)         = default;

  Region &operator=(const Region &) = default;
  Region &operator=(Region &&)      = default;

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
  static Region fromBounds(const Shape &,
                           const std::vector<int64_t> &lower,
                           const std::vector<int64_t> &upper);

  /** Slice in a single dimension. */
  static Region fromBounds(const Shape &, Dimension, uint64_t l, uint64_t u);

  /**
   * Create a new Region based on the Setts of this Region. The new Region
   * has Shape #outShape, and it has Setts at #dimsTo corresponding to the
   * setts in this Region at #dimsFrom. Any dimension not in #dimsTo has the
   * always-on Sett (i.e. no sampling/slicing in those dimensions).
   *
   * outShape[dimsTo[i]] must equal shape(dimsFrom[i]). That is, Setts must
   * migrate to dimensions of the same size.
   *
   * Example: If this region is
   *     (shape=(4,3), setts=(dim0:(1,2,0),dim2:(1,1,0)))
   *
   * Then sampleAtPermutedDims(outShape=(3,7,2), dimsFrom=(1), dimsTo=(0)) is
   *     ((3,7,2), setts=(dim0:(1,1,0),dim1:always-on,dim2:always-on))
   */
  Region sampleAtPermutedDims(const Shape &outShape,
                              const Dimensions &dimsFrom,
                              const Dimensions &dimsTo) const;

  /**
   * Construct a Region with always-on Setts in all dimensions, except in
   * dimension #dim which has a depth-1 Sett defined by #stripe.
   * */
  static Region fromStripe(const Shape &, uint64_t dim, const Stripe &stripe);

  /**
   * Construct a Region which is always on (1) in all dimensions other than
   * dimension #d. In dimension #d, the Region is only on at indices i
   * where (i mod s) = 0.
   *
   * Example. If Shape is (2,8), stride=3 and dim =1, then returned Region is
   *                1..1..1.
   *                1..1..1.
   * \sa fromStripe
   * */
  static Region
  fromStrideAndDim(const Shape &sh, Stride stride, Dimension dim) {
    return fromStripe(sh, dim.get(), {1, stride.get_i64() - 1, 0});
  }

  /**
   * Construct a Region which is only at i in dim(d) if
   *                                    (i mod #strides[d] == 0) */
  static Region fromStrides(const Shape &, const Strides &strides);

  /**
   * Construct a Region which is only at i in Dimension d if
   *                                    (i mod #strides[d] == 0) */
  static Region fromStride(const Shape &, Stride, Dimension);

  /**
   * \return Region which contains all elements of #shape.
   * */
  static Region createFull(const Shape &shape);

  /**
   * \return A Region which contains no elements, contained in volume #shape
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
   * \return The number of elements defined by the Sett along dimension #dim.
   * */
  int64_t nelms(uint64_t dim) const;

  /**
   * \return The number of elements defined by each Setts. The total number of
   *         elements in the Region is the product of these values.
   * */
  std::vector<int64_t> nelms() const;

  /**
   * Recall that a Region is defined as the outer product of independent
   * Setts. Recall too that s Sett defines a 1-d set. For example, the Region
   *
   *       .........      0
   *       11.11.11.   -- 1
   *       .........      2
   *       .........   -- 3
   *       11.11.11.      4
   *
   *       012345678
   *        __ __ __
   *
   * , which contains 12 elements, is defined by a Sett in the vertical
   * dimension, which represents the set (1,4), and a Sett in the
   * horizontal direction, which represents the set (0,1,3,4,6,7). The method
   * therefore returns {{1,4},{0,1,3,4,6,7}}.
   *
   * The method is called "getOns" as it returns the indices which are "on" in
   * each dimension.
   * */
  std::vector<std::vector<int64_t>> getOns() const;

  const std::vector<Sett> &setts() const { return setts_; }

  const Sett &sett(uint64_t d) const { return setts()[d]; }

  /**
   * \return true if the Region contains no elements.
   * */
  bool empty() const;

  /**
   * \return true if this Region contains all elements of containing volume.
   * */
  bool full() const;

  /**
   * \param rhs A Region with the same containing Shape as this.
   *
   * \return The intersection of this Region and #rhs. The returned Regions
   *         have the same containing Shape as this Region.
   * */
  DisjointRegions intersect(const Region &rhs) const;

  /**
   * \return The elementwise complement of this Region. The returned
   *         DisjointRegions and this Region form a partition of the
   *         containing Shape.
   * */
  DisjointRegions getComplement() const;

  /**
   * \param rhs The Region to subtract from this Region.
   *
   * \return All elements in this Region which are not in rhs. It is the
   *         intersection of this Region and the complement of rhs.
   * */
  DisjointRegions subtract(const Region &rhs) const;

  /**
   * A generalization of slicing and sub-sampling.
   *
   * \param where The Region which defines the indices of this Region to
   *              select. It must have the same containing Shape as this.
   *
   * \return DisjointRegions, whose containing Shape is equal to
   *         where.nelms().
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
   * As seen in the example above, all the '1's in #scaffold are replaced
   * by the values in this Region. scaffold.nelms() must equal this Region's
   * containing Shape, in this example this is (3,3).
   * */
  DisjointRegions settFillInto(const Region &scaffold) const;

  /**
   * The inverse of settFillInto, an example is:
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
   * Slice this Region. This method is a convenience function, which calls
   * settSample.
   *
   * \return DisjointRegions of shape #upper - #lower
   * */
  DisjointRegions slice(const std::vector<int64_t> &lower,
                        const std::vector<int64_t> &upper) const;

  /**
   * The inverse operation of expand. Example:
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
  Region dimShuffle(const Permutation &) const;

  /**
   * Expand this Region. This is equivalent to numpy.broadcast_to
   * */
  Region expand(const Shape &to) const;

  /**
   * Attempt to merge this Region with #other. If not possible, the returned
   * object is empty.
   * */
  OptionalRegion merge(const Region &other) const;

  /**
   * Append debug information about this Region, by printing the Setts in each
   * dimension.
   * */
  void append(std::ostream &ss) const;

  /**
   * Append debug information about this Region, by printing the expansion of
   * Setts in each dimension into `1' (on) and `0' (off).
   * */
  void appendBitwise(std::ostream &ss) const;
  std::string getBitwiseString() const;

  /**
   * \return A debug string.
   * */
  std::string str() const;

  /**
   * \return true if rhs and this have an empty intersection.
   * */
  bool disjoint(const Region &rhs) const;

  /**
   * \return true if rhs and this have exactly the same elements and
   *         containing Shape.
   * */
  bool equivalent(const Region &rhs) const;

  /**
   * \return true if all elements in #rhs are also in this, and #rhs has
   *         the same containing Shape as this.
   * */
  bool contains(const Region &rhs) const;

  /**
   * \return true if a and b contain the same elements and have the same
   *         containing Shapes.
   * */
  static bool equivalent(const DisjointRegions &a, const DisjointRegions &b);

private:
  Shape shape_;
  std::vector<Sett> setts_;

  /**
   * Regions with rank-0 Shapes, also called 'scalar' regions, need to be
   * handled separately. To convince you that this is (unfortunately) true,
   * consider the following statement:
   *
   * 'For any Region r,
   *     r.totalElms() + r.getComplement().totalElms() =
   *                                                   = r.shape().nelms()'
   *
   * totalElms is the total number of 'ons' in the region, computed as the
   * product over dimensions of the number of 'ons' in that dimension. The
   * base reduction case, is totalElms = 1 for rank-0 regions. But if used
   * this base case for rank-0 regions, the above statement would not be true
   * (1 + 1 != 1).
   *
   * More generally, it is useful to distinguish between a 'full' and an
   * 'empty' scalar region. This is particularly useful when we're keeping
   * track of how much of a tensor is 'covered': sum over totalElms of the
   * disjoint regions of the tensor's shape. This only works of we can have a
   * scalar region with 0 'ons'.
   * */
  bool fullScalar{true};

  /** Constructor for a scalar region. */
  Region(bool r) : shape_({}), setts_({}), fullScalar(r) {}

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

  /**
   * Create the union of all the Regions in #regs, ensuring that they are
   * disjoint by taking set differences where required.
   * */
  static DisjointRegions
  createUnion(const std::vector<DisjointRegions> &regs);

  DisjointRegions(const DisjointRegions &) = default;
  DisjointRegions(DisjointRegions &&regs)  = default;

  DisjointRegions &operator=(const DisjointRegions &) = default;
  DisjointRegions &operator=(DisjointRegions &&)      = default;

  bool disjoint(const DisjointRegions &rhs) const;

  /**
   * \return The intersection of this DisjointRegions and #rhs.
   *
   * \sa Region::intersect
   * */
  DisjointRegions intersect(const DisjointRegions &rhs) const;

  DisjointRegions(const Region &s) : sh_(s.shape()), regs_({s}) {}

  size_t size() const { return regs_.size(); }

  /**
   * If all Regions are empty, then their union is empty. There is an
   * optimization in the constructor of DisjointRegions to remove empty
   * Regions, so this should be equivalent to checking that regs_ is empty. In
   * case there is a future constructor which allows empty Regions in regs_
   * however, we prefer to explicily count all elements in all Regions.
   * */
  bool empty() const { return totalElms() == 0; }

  const std::vector<Region> &get() const { return regs_; }

  const Region &at(size_t i) const { return regs_[i]; }

  /**
   * \return true if the Regions are mutually disjoint and have the same
   *         containing Shape.
   * */
  bool isValid() const;
  void confirmValid() const;

  int64_t totalElms() const;

  bool full() const { return totalElms() == shape().nelms(); }

  /** Append r to regs_  */
  void insert(const Region &r);

  //  The following methods are the vector extensions of their corresponding
  //  single Region versions.

  std::vector<int64_t> nelms() const;

  DisjointRegions flatten() const;

  DisjointRegions subtract(const DisjointRegions &) const;

  DisjointRegions reduce(const Shape &) const;

  DisjointRegions expand(const Shape &) const;

  DisjointRegions slice(const std::vector<int64_t> &lower,
                        const std::vector<int64_t> &upper) const;

  DisjointRegions settFillInto(const Region &) const;

  DisjointRegions settSample(const Region &where) const;

  DisjointRegions reverse(const std::vector<uint64_t> &dimensions) const;

  DisjointRegions reshape(const Shape &) const;

  DisjointRegions dimShuffle(const Permutation &) const;

  DisjointRegions getComplement() const;

  DisjointSetts flattenToSetts() const;

  bool equivalent(const DisjointRegions &rhs) const {
    return Region::equivalent(*this, rhs);
  }

  bool equivalent(const Region &rhs) const {
    return equivalent(DisjointRegions(rhs));
  }

  /**
   * \return true if all elements in all Regions in #rhs are also in one of
   *         this DisjointRegions' Regions.
   *
   * \sa Region::contains
   * */
  bool contains(const DisjointRegions &rhs) const;

  /**
   * Append debug information.
   * */
  void append(std::ostream &ss) const;
  void appendBitwise(std::ostream &ss) const;
};

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
