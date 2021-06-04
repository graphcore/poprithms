// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_ORIGINS_HPP
#define POPRITHMS_MEMORY_ALIAS_ORIGINS_HPP

#include <map>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace memory {
namespace alias {

using ndarray::Shape;
using nest::DisjointRegions;
using nest::Region;

/** A class to represent the allocations that a Tensor is composed of.
 *
 * An example is, if
 *   A is a shape=(2,2) allocation Tensor, and
 *   B is a shape=(2,2) allocation Tensor, and
 *   C is concat({A.slice((0,0),(1,2)), B}, axis = 0)
 *   so that C is a shape=(3,2) Tensor.
 *
 * The Origins for C will represent the regions in the origin allocations
 * A and B which C used. In particular, the member oMap stores:
 *
 *   {"A" : {"the full (2,2) Region"}, "B" : {"a (1, 2) slice"} }.
 *
 */
class Origins {
public:
  /**
   * \param sh The Shape of the Tensor whose origins are being represented.
   * */
  Origins(const Shape &sh) : shape(sh) {}

  Origins(Origins &&)      = default;
  Origins(const Origins &) = default;
  Origins &operator=(Origins &&) = default;
  Origins &operator=(const Origins &) = default;

  /** Register an allocation.
   *
   * \param id The unique identifier of the allocation.
   *
   * \param regs Regions of the source allocation Tensor which are aliased.
   *             Note that the DisjointRegions' shape is not necessarily the
   *             same as this Origins' shape.
   * */
  void insert(AllocId, const DisjointRegions &regs);

  /** Append all of the registered allocations of rhs */
  void insert(const Origins &rhs);

  /** Return the AllocIds which this Origins has at least 1 element of. */
  std::vector<AllocId> getAllocIds() const;

  std::unique_ptr<Origins> clone() const;

  const std::vector<DisjointRegions> &at(AllocId id) const {
    return oMap.at(id);
  }

  /** \return true iff there are any duplicated allocation addresses */
  bool containsAliases() const;

  /** \return true iff this Origins has at least 1 allocation address in
   * common with rhs */
  bool isAliasedTo(const Origins &rhs) const;

  /** Slightly weaker than poplar's isContiguous, this method returns true iff
   * there is some permutation of the  allocation addresses registered for
   * which poplar's isContiguous return true.
   *
   * \return true if
   *          1) oMap has more than 1 key,
   *          2) containsAliases() is false,
   *          3) the elements form a row-major contiguous set
   *
   * */
  bool isRowMajorSetContiguous() const;

  void append(std::ostream &) const;

  void clear() {
    oMap.clear();
    sumTotalRegionSizes = 0;
  }

  /** map all keys `k' in oMap, to crt[k]. */
  Origins remap(const std::vector<uint64_t> &crt) const;

private:
  // A map from AllocId to DisjointRegions. Design decision: We could have
  // std::map<AllocId, DisjointRegions>, that is, a single DisjointRegions
  // instead of a vector of them. Using this non-vector approach would require
  // subtracting DisjointRegions and only inserting the novel elements. For
  // many uses of Origins, this is unnecessary and the required information
  // can be obtained without doing the subtraction. Thus taking this lazy/jit
  // approach of keeping a vector of DisjointRegions, the union of which
  // represents all the addresses of the allocation (the key) aliased.
  std::map<AllocId, std::vector<DisjointRegions>> oMap;

  Shape shape;
  uint64_t sumTotalRegionSizes{0};
  void incrementSumTotalRegionSizes(uint64_t);
};

std::ostream &operator<<(std::ostream &, const Origins &);

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
