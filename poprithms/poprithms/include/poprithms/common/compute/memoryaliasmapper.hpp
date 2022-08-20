// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_MEMORYALIASMAPPER_HPP
#define POPRITHMS_COMMON_COMPUTE_MEMORYALIASMAPPER_HPP

#include <poprithms/common/multiout/fwdedgemap.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/additionalfwdedges.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/alias/mapper.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Graph;

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;

static const poprithms::memory::alias::Color MemoryAliasConstant(0);
static const poprithms::memory::alias::Color MemoryAliasVariable(1);

/**
 * A mapping between tensors in a common::compute::Graph, and tensors in a
 * memory::alias::Graph. The mapping can be grown incrementally.
 * */
class MemoryAliasMapper : public poprithms::memory::alias::Mapper<TensorId> {
public:
  /**
   * Construct a mapping that includes all of the tensors #tIds in #g. Other
   * tensors in #g might be added too, for example all the variable tensors of
   * that the tensors #tIds are composed of.
   * */
  MemoryAliasMapper(const Graph &g, const TensorIds &tIds);

  /**
   * Extend the mapping to include the tensors #tIds in the compute::Graph. If
   * the tensors are already present in the mapping, then nothing is added.
   * */
  void extend(const TensorIds &);

  std::string external() const final;

  /**
   * \return All tensors that are aliased to a tensor in #tIds. This
   *         includes all tensors in #tIds which have at least 1 element.
   *
   * Computing this set of tensors is done in 2 steps.
   *
   * 1) Find all tensors which \b might be aliased to a tensor in #tIds. This
   *    is done by traversing the graph through all aliasing edges of ops.
   *    This MemoryAliasMapper is then extended to include all these found
   *    tensors.
   *
   * 2) Perform accurate alias analysis to find the subset of (1) which are
   *    truly aliased.
   *
   * As an example to show that step (2) is required, consider the following:
   * <code>
   *   auto c = concat_({b, a},0).slice_(Dimension(0),0,1);
   * </code>
   *
   * In this case #c is not aliased to #a, although the backtracking algorithm
   * in (1) will traverse through it. Step (2) will remove a.
   * */
  TensorIds aliases(const TensorIds &tIds) {
    extend(potentialMultiGraphAliases(computeGraph, tIds));
    return aliasesFromExtended(tIds);
  }

private:
  const Graph &computeGraph;

  /**
   * A set of tensors which is guaranteed to contain all aliases, across all
   * sub-graphs, of aliases of the tensors in #tIds.
   * */
  static TensorIds potentialMultiGraphAliases(const Graph &g,
                                              const TensorIds &tIds);

  /**
   * All of the aliases, in all sub-graphs, of the tensors in #tIds.
   * This method can only be called when it is known that all aliases of #tIds
   * are known to be in this MemoryAliasMapper.
   * */
  TensorIds aliasesFromExtended(const TensorIds &tIds) const;
};

/**
 * Utility class for querying alias related information about a
 * compute::Graph.
 * */
class AliasGraphQuerier {

public:
  /**
   * \return true if all of the allocations that #tId is composed of are
   *         (1) constant and (2) zero. The tensor #tId must belong to the
   *         graph #g.
   * */
  static bool isAllConstZero(const Graph &g, const TensorId &tId);

  /**
   * \return The set of constraints required between ops to ensure that ops
   *         which modify their inputs do so after any other op which uses the
   *         modifier's input, or alias thereof. The edges returned are
   *         forward constraints (map key before value).
   *
   * Example 1:
   * ---------
   *      +--> relu_
   * t0  -+
   *      +--> sin
   *
   * The constraint sin -> relu_ is returned: relu_ modifies t0 and sin
   * consumes t0, so relu_ must be scheduled after sin.
   *
   *
   * Example 2:
   * ---------
   * t0 -> relu_ -> cos_
   *
   * No constraints are returned  as the order between relu_ and cos_ is
   * defined by the data (tensor) dependency.
   *
   *
   *
   * Example 3:
   * ---------
   *      +--> relu_
   * t0  -+
   *      +--> sin_
   *
   * The constraints relu_ -> sin_ and sin_ -> relu_ are both returned (a
   * cycle).
   *
   *
   *
   * */
  static std::map<OpId, OpIds>
  makeModifiersFinalConsumers(const Graph &, const OpIds &requiredKeys);
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
