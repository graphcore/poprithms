// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>
#include <util/error.hpp>

#include <poprithms/util/stridedpartition.hpp>

namespace poprithms {
namespace util {

StridedPartition::StridedPartition(uint64_t range,
                                   uint64_t groupSize,
                                   uint64_t stride)
    : range_(range), groupSize_(groupSize), stride_(stride) {

  auto getBase = [&]() {
    std::ostringstream oss;
    std::string nl{"\n     "};
    oss << "Invalid StridedPartition : " << nl //
        << "range      = " << range << nl      //
        << "group size = " << groupSize << nl  //
        << "stride     = " << stride << ".";
    return oss.str();
  };

  if (range == 0 || groupSize == 0 || stride == 0) {
    throw error(getBase() + "\nAll parameters must be strictly positive.");
  }

  if (range < groupSize) {
    throw error(getBase() +
                "\nNumber of indices (range) < indices per group.");
  }

  auto packSize_ = stride * groupSize;
  if (range_ % packSize_ != 0) {
    throw error(getBase() +
                "\nNumber of indices % (stride * indices per group) != 0.");
  }
}

std::vector<uint64_t> StridedPartition::indicesInGroup(uint64_t rg) const {
  std::vector<uint64_t> vs;
  vs.reserve(groupSize());
  auto packId  = (rg / groupsPerPack());
  auto offset0 = indicesPerPack() * packId;
  for (uint64_t i = 0; i < groupSize(); ++i) {
    vs.push_back(offset0 + stride_ * i + rg % groupsPerPack());
  }
  return vs;
}

std::vector<std::vector<uint64_t>> StridedPartition::groups() const {
  std::vector<std::vector<uint64_t>> gs;
  gs.reserve(nGroups());
  for (uint64_t i = 0; i < nGroups(); ++i) {
    gs.push_back(indicesInGroup(i));
  }
  return gs;
}

void StridedPartition::append(std::ostream &ost) const {
  ost << "(range=" << range() << ",groupSize=" << groupSize()
      << ",stride=" << stride() << ")";
}

uint64_t StridedPartition::group(uint64_t index) const {
  return pack(index) * groupsPerPack() + index % stride_;
}

uint64_t StridedPartition::firstInGroup(uint64_t group) const {

  // Running example:
  // replication factor = 8, group size = 2, stride = 2.
  //
  // This method can answer: What is the first index ("replica") in group #3?
  //
  // 01234567 index ("replica")
  //      ^
  //      |
  // 01012323 group
  //      =
  //
  // The first appearance of 3 in the group listing is for index=5. So this
  // method will return firstInGroup(3) = 5.
  //

  // A "pack" is a set of interleaved groups. There are stride_ groups in 1
  // pack. The pack to which #group belongs is:
  //
  // In the running example this is the pack "2323" which is pack number 1
  // (pack number zero is "0101").
  auto packId = group / stride_;

  // The number of indices in 1 pack.
  //
  // In the running example this is 4 (there are 4 indices in "0101" and
  // "2323").
  auto packSize = stride_ * groupSize_;

  // The index at which the pack of #group starts.
  //
  // In the running example this is 4, where "2323" starts.
  auto packStart = packId * packSize;

  // The index of the first group of this pack:
  //
  // In the running example this is 2, the first index in "2323".
  auto firstGroupInPack = packId * stride_;

  // In the running example this is 3 - 2 = 1.
  auto offset = group - firstGroupInPack;

  // In the running example this is 5, as expected.
  return packStart + offset;
}

} // namespace util
} // namespace poprithms
