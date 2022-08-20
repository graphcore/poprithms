// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <common/compute/error.hpp>

#include <poprithms/common/compute/simtensormap.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

void SimTensorMap::copy(const TensorId &src, const TensorId &dst) const {

  auto srcTensors = getValue(src);
  auto dstTensors = getValue(dst);

  if (srcTensors.size() != dstTensors.size()) {
    std::ostringstream oss;
    oss << "Failure in copy, src has " << srcTensors.size()
        << " host tensors, but dst has " << dstTensors.size()
        << " host tensors. ";
    throw error(oss.str());
    for (uint64_t r = 0; r < srcTensors.size(); ++r) {
      dstTensors[r].update_(srcTensors[r]);
    }
  }
}

void SimTensorMap::copy(const TensorIds &srcs, const TensorIds &dsts) const {

  if (srcs.size() != dsts.size()) {
    std::ostringstream oss;
    oss << "Error in copy(srcs=" << srcs << ", dsts=" << dsts << "). "
        << "Expected srcs and dsts to have the same number of tensors, but "
        << srcs.size() << " != " << dsts.size() << '.';
    throw error(oss.str());
  }

  for (uint64_t i = 0; i < srcs.size(); ++i) {
    copy(srcs[i], dsts[i]);
  }
}

uint64_t
SimTensorMap::getNTensorsByUnanimity(const TensorIds &tensorIds) const {

  if (tensorIds.empty()) {
    throw error("Cannot ascertain the common number of host tensors from "
                "empty set of tensors. ");
  }

  std::vector<uint64_t> factors;
  factors.reserve(tensorIds.size());
  for (const auto &tId : tensorIds) {
    factors.push_back(getValue(tId).size());
  }

  const auto F0 = factors[0];
  if (std::any_of(factors.cbegin(), factors.cend(), [F0](auto x) {
        return x != F0;
      })) {
    std::ostringstream oss;
    oss << "Failed to ascertain a common number of host tensors for "
        << tensorIds << ", which have numbers of host tensors ";
    poprithms::util::append(oss, factors);
    oss << " respectively: no unanimity. ";
    throw error(oss.str());
  }

  return F0;
}

HostTensors SimTensorMap::getTensors(const TensorIds &ids, uint64_t r) const {
  HostTensors ts;
  ts.reserve(ids.size());
  for (auto id : ids) {
    auto &&allReplicas = getValue(id);
    if (allReplicas.size() <= r) {
      std::ostringstream oss;
      oss << "Failed to retrieve replica #" << r << " for tensor " << id
          << ". This SimTensorMap only has " << allReplicas.size()
          << " replica(s) for this tensor.";
      throw error(oss.str());
    }

    ts.push_back(allReplicas[r]);
  }
  return ts;
}

std::unique_ptr<SimTensorMap> SimTensorMap::clone() const {
  return std::make_unique<SimTensorMap>(*this);
}

void SimTensorMap::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

} // namespace compute
} // namespace common

} // namespace poprithms
