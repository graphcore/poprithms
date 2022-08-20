// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/guide/graphinfo.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

TensorId GraphInfo::inTensorId(const OpTraversal &ot) const {
  return inTensorId(ot.opId(), ot.inIndex());
}

bool GraphInfo::gradientPropagates(const TensorId &id) const {
  for (uint64_t i = 0; i < nInTensors(id.opId()); ++i) {
    if (gradientPropagates(
            OpTraversal{InIndex(i), id.opId(), id.outIndex()})) {
      return true;
    }
  }
  return false;
}

TensorIds GraphInfo::outTensorIds(OpId opId) const {
  TensorIds outIds;
  const auto nOuts = nOutTensors(opId);
  outIds.reserve(nOuts);
  for (uint64_t o = 0; o < nOuts; ++o) {
    outIds.push_back({opId, o});
  }
  return outIds;
}

void GraphInfo::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

} // namespace guide
} // namespace autodiff
} // namespace poprithms
