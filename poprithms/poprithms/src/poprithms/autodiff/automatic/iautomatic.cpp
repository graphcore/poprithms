// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>
#include <poprithms/autodiff/core/autodiff.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

SubGraphId
IAutomaticQuerier::subGraphIdFromTensorIds(const TensorIds &tIds) const {
  return SubGraphId::fromTensorIds(*this, tIds);
}

SubGraphIds IAutomaticQuerier::callees(OpId opId) const {
  SubGraphIds sgIds;
  const auto N = nCallees(opId);
  sgIds.reserve(N);
  for (uint64_t i = 0; i < N; ++i) {
    sgIds.push_back(callee(opId, CalleeIndex(i)));
  }
  return sgIds;
}

TensorIds IAutomaticQuerier::inTensorIds(OpId opId) const {
  const uint64_t N = nInTensors(opId);
  TensorIds tIds;
  tIds.reserve(N);
  for (InIndex i = 0; i < N; ++i) {
    tIds.push_back(inTensorId(opId, i));
  }
  return tIds;
}

TensorIds IAutomaticQuerier::inDsts(OpId opId,
                                    const InIndices &inIndices) const {
  TensorIds tIds;
  tIds.reserve(inIndices.size());
  for (auto i : inIndices) {
    tIds.push_back(inDst(opId, i).tId());
  }
  return tIds;
}

TensorIds IAutomaticQuerier::outTensorIds(OpId opId) const {
  TensorIds tIds;
  tIds.reserve(nOutTensors(opId));
  for (OutIndex o = 0; o < nOutTensors(opId); ++o) {
    tIds.push_back({opId, o});
  }
  return tIds;
}

TensorId IAutomaticMutator::zeroLike(const TensorId &tId,
                                     SubGraphId sgId,
                                     const std::string &n) {
  auto x = scalarConstantLike(tId, sgId, 0, n);
  x      = expand_(x, shape(tId));
  return x;
}

SubGraphId
IAutomaticQuerier::subGraphIdFromObjective(const Objective &o) const {
  return subGraphIdFromTensorIds(o.allTensorIds());
}

SubGraphId IAutomaticQuerier::subGraphIdFromTensorIds(
    const std::vector<TensorIds> &tIdss) const {
  return subGraphIdFromTensorIds(TensorId::flatten(tIdss));
}

TensorIds IAutomaticQuerier::outSources(OpId opId,
                                        CalleeIndex ci,
                                        const OutIndices &outIndices) const {
  TensorIds ids;
  ids.reserve(outIndices.size());
  for (auto o : outIndices) {
    ids.push_back(outSource(opId, o, ci));
  }
  return ids;
}

} // namespace automatic
} // namespace autodiff
} // namespace poprithms
