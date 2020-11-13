// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/inplace/tensorid.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

void TensorMap::insert(const TensorId &tensorId,
                       alias::TensorId aliasTensorId) {

  const auto opId_        = tensorId.opId();
  const auto opId_u64     = static_cast<uint64_t>(opId_.get());
  const auto outIndex_    = tensorId.outIndex();
  const auto outIndex_u64 = static_cast<uint64_t>(outIndex_.get());

  if (toAliasGraph.size() <= opId_u64) {
    toAliasGraph.resize(opId_u64 + 1);
  }
  if (toAliasGraph[opId_u64].size() <= outIndex_u64) {
    toAliasGraph[opId_u64].resize(outIndex_u64 + 1);
  }
  toAliasGraph[opId_u64][outIndex_u64] = aliasTensorId;

  const auto aliasTensorId_u64 = static_cast<uint64_t>(aliasTensorId.get());

  if (fromAliasGraph.size() <= aliasTensorId_u64) {
    fromAliasGraph.resize(aliasTensorId_u64 + 1);
  }
  fromAliasGraph[aliasTensorId_u64] = tensorId;
}

TensorIds TensorMap::fromAliasGraphIds(
    const std::vector<alias::TensorId> &aliasGraphIds) const {
  TensorIds ids;
  ids.reserve(aliasGraphIds.size());
  for (const auto aliasGraphId : aliasGraphIds) {
    ids.push_back(fromAliasGraph[aliasGraphId.get()]);
  }
  return ids;
}

alias::TensorId TensorMap::toAliasGraphId(const TensorId &id) const {
  return toAliasGraph[id.opId().get()][id.outIndex().get()];
}

std::vector<alias::TensorId>
TensorMap::toAliasGraphIds(const TensorIds &ids) const {
  std::vector<alias::TensorId> aliasIds;
  aliasIds.reserve(ids.size());
  for (const auto &id : ids) {
    aliasIds.push_back(toAliasGraphId(id));
  }
  return aliasIds;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
