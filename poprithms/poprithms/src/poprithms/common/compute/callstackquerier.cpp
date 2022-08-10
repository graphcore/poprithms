// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <set>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/callstackquerier.hpp>
#include <poprithms/common/compute/prune/pruner.hpp>

namespace poprithms {
namespace common {
namespace compute {

InIndices CallstackQuerier::nonCalleeCopyInIndices(OpId opId) const {
  if (!graph().computeOp(opId).hasCallees()) {
    return graph().computeOp(opId).inIndices();
  }
  return wc(opId).nonCopyToCalleeIndices();
}

std::vector<std::pair<InIndex, TensorId>>
CallstackQuerier::copyInDsts(OpId opId) const {
  auto &&tIds = wc(opId).inTensorIdDsts();
  std::vector<std::pair<InIndex, TensorId>> v;
  v.reserve(tIds.size());
  for (uint64_t i = 0; i < tIds.size(); ++i) {
    v.push_back({i, tIds[i]});
  }
  return v;
}

bool CallstackQuerier::isCarriedFrom(const TensorId &tId,
                                     const CallStack &cs) const {
  // empty call stack means this is not inside a repeat, and so is not carried
  // to:
  if (cs.empty()) {
    return false;
  }
  return wc(cs.back().caller()).isCarriedFrom(tId);
}

bool CallstackQuerier::isCarriedTo(const TensorId &tId,
                                   const CallStack &cs) const {
  // empty call stack means this is not inside a repeat, and so is not carried
  // to:
  if (cs.empty()) {
    return false;
  }
  return wc(cs.back().caller()).isCarriedTo(tId);
}

TensorId CallstackQuerier::carriedFrom(const TensorId &tId,
                                       const CallStack &cs) const {
  auto caller = cs.back().caller();
  return wc(caller).carriedFrom(tId);
}

TensorId CallstackQuerier::carriedTo(const TensorId &tId,
                                     const CallStack &cs) const {
  auto caller = cs.back().caller();
  return wc(caller).carriedTo(tId);
}

} // namespace compute
} // namespace common
} // namespace poprithms
