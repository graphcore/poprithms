// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_TENSORMAP_HPP
#define POPRITHMS_MEMORY_INPLACE_TENSORMAP_HPP

#include <poprithms/memory/inplace/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/** Map TensorIds between
 * 1) poprithms::memory::inplace::Graph, and
 * 2) poprithms::memory::alias::Graph.
 * */
struct TensorMap {
  ToAliasGraph toAliasGraph;
  FromAliasGraph fromAliasGraph;
  void insert(const TensorId &, alias::TensorId);

  alias::TensorId toAliasGraphId(const TensorId &) const;
  std::vector<alias::TensorId> toAliasGraphIds(const TensorIds &) const;

  TensorId fromAliasGraphId(alias::TensorId) const;
  TensorIds fromAliasGraphIds(const std::vector<alias::TensorId> &) const;
};

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
