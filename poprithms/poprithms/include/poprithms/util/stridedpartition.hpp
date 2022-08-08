// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_STRIDEDPARTITION_HPP
#define POPRITHMS_UTIL_STRIDEDPARTITION_HPP

#include <array>
#include <ostream>
#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace util {

/**
 * Partition the interval [0, range) into equally sized groups, where the
 * indices in each group are separated by a constant stride.
 *
 * Example 1:
 *
 * range      = 6
 * group size = 3
 * stride     = 1
 *
 * index    :   0 1 2 3 4 5
 *              -----------
 * group id :   0 0 0 1 1 1 (2 groups of 3 elements).
 *
 * Example 2:
 *
 * range      = 18
 * group size = 3
 * stride     = 2
 *
 * The indices in [0, 18) are divided into 6 groups of 3 indices (elements).
 * The indices in each group are separated by a distance of stride=2:
 *
 * index:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
 *           -----------------------------------------------------
 * group id: 0  1  0  1  0  1  2  3  2  3  2  3  4  5  4  5  4  5
 *           <====>                  <=====>                      (stride=2)
 *           0     0     0                                    (group size=3)
 *              1     1     1
 *                             2     2     2
 *                                3     3     3
 *                                               4     4     4
 *                                                  5     5     5
 * */
class StridedPartition {
public:
  /**
   * See the example above for definitions of the arguments.
   *
   * #range must be divisible by #stride * #groupSize.
   * */
  StridedPartition(uint64_t range, uint64_t groupSize, uint64_t stride);

  bool operator==(const StridedPartition &rhs) const {
    return atts() == rhs.atts();
  }
  bool operator!=(const StridedPartition &x) const { return !operator==(x); }

  /**
   * The total number of indices being partitioned. The indices are
   * contiguous, starting from 0.
   * */
  uint64_t range() const { return range_; }

  /**
   * The size of each of the groups.
   * */
  uint64_t groupSize() const { return groupSize_; }

  uint64_t nGroups() const { return range() / groupSize(); }

  /**
   * The distance between consecutive indices in a group.
   * */
  uint64_t stride() const { return stride_; }

  /**
   * The group to which #index \in [0, range) belongs.
   * */
  uint64_t group(uint64_t index) const;

  /**
   * The full partition of indices into groups. The returned vector contains
   * #nGroups() vectors. The #i'th sub-vector contains the indices in group
   * #i.
   * */
  std::vector<std::vector<uint64_t>> groups() const;

  /**
   * \return All the indices in the group #group.
   * */
  std::vector<uint64_t> indicesInGroup(uint64_t group) const;

  void append(std::ostream &) const;

private:
  std::tuple<uint64_t, uint64_t, uint64_t> atts() const {
    return {groupSize_, stride_, range_};
  }

  uint64_t range_;
  uint64_t groupSize_;
  uint64_t stride_;

  // The concept of a 'pack' is not needed in the public facing API but makes
  // calculations here easier. A 'pack' is a collection of interleaved groups.
  // Continuing the example presented at the start of the class:
  //
  //  index:      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
  //              -----------------------------------------------------
  //  group id :  0  1  0  1  0  1  2  3  2  3  2  3  4  5  4  5  4  5
  //  pack id  :  0  0  0  0  0  0  1  1  1  1  1  1  2  2  2  2  2  2
  //
  //
  // Specifically,
  //
  // number of groups per pack  = stride
  // number of indices per pack = stride * indices per group
  // number of packs            = number of indices / indices per pack
  //
  uint64_t groupsPerPack() const { return stride_; }
  uint64_t indicesPerPack() const { return groupSize_ * groupsPerPack(); }
  uint64_t nPacks() const { return nGroups() / indicesPerPack(); }
  uint64_t pack(uint64_t index) const { return index / indicesPerPack(); }
};

inline std::ostream &operator<<(std::ostream &ost,
                                const StridedPartition &sp) {
  sp.append(ost);
  return ost;
}

} // namespace util
} // namespace poprithms

#endif
