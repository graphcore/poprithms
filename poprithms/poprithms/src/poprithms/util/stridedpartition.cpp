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

} // namespace util
} // namespace poprithms
