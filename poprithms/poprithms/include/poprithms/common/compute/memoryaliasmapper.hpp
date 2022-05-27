// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_MEMORYALIASMAPPER_HPP
#define POPRITHMS_COMMON_COMPUTE_MEMORYALIASMAPPER_HPP

#include <string>

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/mapper.hpp>
#include <poprithms/memory/alias/node.hpp>
#include <poprithms/memory/alias/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Graph;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;

static const poprithms::memory::alias::Color MemoryAliasConstant(0);
static const poprithms::memory::alias::Color MemoryAliasVariable(1);

/**
 * Mapping between tensors in a common::compute::Graph and tensors in a
 * memory::alias::Graph.
 * */
class MemoryAliasMapper : public poprithms::memory::alias::Mapper<TensorId> {
public:
  /**
   * Construct a mapping which includes all the tensors #tIds in #g. Other
   * tensors in #g might be added too, for example all the variable tensors of
   * which the #tIds are composed.
   * */
  MemoryAliasMapper(const Graph &g, const TensorIds &tIds);

  /**
   * Extend the mapping to include the tensors #tIds in the compute::Graph. If
   * the tensors are already present in the mapping then nothing is added.
   * */
  void extend(const TensorIds &);

  std::string external() const final { return "machine"; }

private:
  const Graph &graph_;
};

class AliasQuerier {
public:
  /**
   * \return true if all elements of the tensor #tId are constant and zero.
   * */
  static bool isAllConstZero(const Graph &g, const TensorId &tId);
};

} // namespace compute
} // namespace common
} // namespace poprithms
#endif
