// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <map>
#include <set>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/memoryaliasmapper.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/compute/host/regionutil.hpp>
#include <poprithms/memory/alias/jitgrower.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::compute::MemoryAliasMapper;
using common::multiout::FwdEdgeMap;
using common::multiout::InIndex;
using common::multiout::InIndices;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using common::schedulable::SubGraphId;

std::string MemoryAliasMapper::external() const {
  return "poprithms::common::compute";
}

namespace {
/**
 * Just-in-time helper class for adding (growing) only the tensors necessary,
 * when necessary.
 * */
class JitAliasGrower : public memory::alias::JitGrower<TensorId> {
private:
  const Graph &g;
  MemoryAliasMapper &mam;

public:
  JitAliasGrower(const Graph &g_, MemoryAliasMapper &mam_)
      : g(g_), mam(mam_) {}

  /**
   * All aliasing inputs of the op which creates #nxt, as well as the root
   * reference of #nxt.
   * */
  TensorIds aliasingIns(const TensorId &nxt) const final {
    const auto nxtOpId = nxt.opId();
    TensorIds aliIns;
    for (InIndex i = 0; i < g.nInTensors(nxtOpId); ++i) {
      if (g.aliases(nxtOpId, i, nxt.outIndex())) {
        aliIns.push_back(g.inTensorId(nxtOpId, i));
      }
    }

    if (g.rootRef(nxt) != nxt) {
      aliIns.push_back(g.rootRef(nxt));
    }
    return aliIns;
  }

  void growAliasTensors(const TensorIds &tensorSchedule) final {
    for (auto &&tId : tensorSchedule) {
      if (!mam.has(tId)) {
        g.computeOp(tId.opId()).growAliasMapper(mam);
      }
    }
  }

  bool containsAliasTensor(const TensorId &tId) const final {
    return mam.has(tId);
  }
};

} // namespace

void MemoryAliasMapper::extend(const TensorIds &tIds) {
  JitAliasGrower jag(computeGraph, *this);
  jag.extend(tIds);
}

MemoryAliasMapper::MemoryAliasMapper(const Graph &g, const TensorIds &tIds)
    : memory::alias::Mapper<TensorId>(), computeGraph(g) {
  extend(tIds);
}

namespace {

class AliasNeighborGetterWithoutRefTraversal {
private:
  const Graph &g;

public:
  AliasNeighborGetterWithoutRefTraversal(const Graph &g_) : g(g_) {}

  TensorIds neighbors(const TensorId &currentTensor) const {
    return get(g, currentTensor);
  };

  // Neighbors of #currentTensor are any of the following tensors that might
  // be aliases:
  //
  // 1) Inputs.
  // 2) Outputs of ops which have #currentTensor as an input. In other words,
  //    outputs of consumers.
  static TensorIds get(const Graph &g, const TensorId &currentTensor) {

    TensorIds neighborsOfCurrent;

    const auto &producerOp = g.computeOp(currentTensor.opId());

    // 1) Inputs that might be aliases:
    for (InIndex i = 0; i < producerOp.nInTensors(); ++i) {
      if (producerOp.aliases(i, currentTensor.outIndex())) {
      }
      neighborsOfCurrent.push_back(producerOp.inTensorId(i));
    }

    // 2) Outputs of consumers that might be aliases:
    for (auto cId : g.consumptionIds(currentTensor)) {
      for (OutIndex o = 0; o < g.nOutTensors(cId.opId()); ++o) {
        if (g.aliases(cId.opId(), cId.inIndex(), o)) {
          neighborsOfCurrent.push_back({cId.opId(), o});
        }
      }
    }
    return neighborsOfCurrent;
  }
};

class AliasNeighborGetterWithRefTraversal {
private:
  const Graph &g;

public:
  AliasNeighborGetterWithRefTraversal(const Graph &g_) : g(g_) {}

  TensorIds neighbors(const TensorId &currentTensor) const {

    // 1) References in different graphs:
    auto v = g.refsExcludingSelf(currentTensor);

    // 2,3) Inputs and outputs that might be aliases:
    auto w = AliasNeighborGetterWithoutRefTraversal::get(g, currentTensor);

    w.insert(w.end(), v.cbegin(), v.cend());
    return w;
  }
};
} // namespace

TensorIds
MemoryAliasMapper::potentialMultiGraphAliases(const Graph &g,
                                              const TensorIds &tIds) {
  return poprithms::common::multiout::depthFirst(
      AliasNeighborGetterWithRefTraversal(g), tIds, [](const TensorId &) {
        return true;
      });
}

bool AliasGraphQuerier::isAllConstZero(const Graph &computeGraph,
                                       const TensorId &tId) {

  MemoryAliasMapper mam(computeGraph, {tId});

  // If any of the allocations of #tId are non-constant and not empty,
  // return false.
  if (mam.graph().containsColor(mam.id(tId), MemoryAliasVariable) &&
      computeGraph.nelms(tId) != 0) {
    return false;
  }

  const auto aliases =
      mam.idsFromAliasIds(mam.graph().allAliases(mam.id(tId)));

  OpIds allocs;

  // For all the aliases of #tId which are constant initializers, check that
  // the elements of the regions used by #tId are all 0.
  for (auto a : aliases) {
    if (computeGraph.isConstInit(a.opId())) {
      auto &&regs = mam.graph().allocRegions(mam.id(tId), mam.id(a));
      for (const auto &regs_ : regs) {
        for (const auto &r : regs_.get()) {
          if (!poprithms::compute::host::RegionUtil::allZero(
                  computeGraph.constInitValue(a.opId()), r)) {
            return false;
          }
        }
      }
      allocs.push_back(a.opId());
    }
  }

  return true;
}

// TODO(T64447) add tests
std::map<OpId, OpIds>
AliasGraphQuerier::makeModifiersFinalConsumers(const Graph &computeGraph,
                                               const OpIds &required) {

  std::map<OpId, OpIds> fwdEdges;

  // Determine which sub-graphs are required from the set of required ops.
  std::set<SubGraphId> subGraphs;
  for (auto opId : required) {
    subGraphs.insert(computeGraph.subGraphId(opId));
  }

  // Process each of the sub-graphs independently.
  for (auto sgId : subGraphs) {

    // The tensors which are consumed by an op which modifies them:
    auto modified = computeGraph.modified(sgId);

    auto aliasedToModified = poprithms::common::multiout::depthFirst(
        AliasNeighborGetterWithoutRefTraversal(computeGraph),
        modified,
        [](const TensorId &) { return true; });

    MemoryAliasMapper mam(computeGraph, aliasedToModified);

    // data dependencies (tensor -> op -> tensor).
    auto fwdEdgeMap = computeGraph.getMultioutForwardEdgeMap_u64(
        computeGraph.opIds(aliasedToModified));

    // O(nOps^2) transitive closure.
    schedule::transitiveclosure::TransitiveClosure tcm(
        fwdEdgeMap.fwdEdgesCompact());

    for (auto &&mId : modified) {
      for (auto c : computeGraph.consumptionIds(mId)) {
        auto opId = c.opId();
        auto i    = c.inIndex();

        if (computeGraph.modifies(opId, i)) {

          /// All of the compute graph tensors in the sub-graph #sgId that
          /// are aliased to the input #i of op #opId (i.e. #mId):
          std::vector<TensorId> aliases;
          for (auto atId : mam.graph().allAliases(mam.id(mId))) {
            if (mam.hasAliasId(atId)) {
              auto ctId = mam.idFromAliasId(atId);
              if (computeGraph.subGraphId(ctId.opId()) ==
                  computeGraph.subGraphId(opId)) {
                aliases.push_back(ctId);
              }
            }
          }

          /// Consumers of aliases.
          OpIds aliasConsumers;
          for (auto tId : aliases) {
            for (const auto &cId : computeGraph.consumptionIds(tId)) {
              if (cId.opId() != opId) {
                if (!computeGraph.computeOp(cId.opId()).isInitializingOp()) {
                  aliasConsumers.push_back(cId.opId());
                }
              }
            }
          }

          /// Ensure consumers of aliases happen before modifier, if there is
          /// any ambiguity in their relative order:
          for (auto cId : aliasConsumers) {
            if (tcm.unconstrainedInBothDirections(
                    fwdEdgeMap.compactId(cId), fwdEdgeMap.compactId(opId))) {

              auto found = fwdEdges.find(cId);
              if (found == fwdEdges.cend()) {
                fwdEdges.insert({cId, {opId}});
              } else {
                found->second.push_back(opId);
              }
            }
          }
        }
      }
    }
  }
  return fwdEdges;
}

TensorIds
MemoryAliasMapper::aliasesFromExtended(const TensorIds &tIds) const {
  std::set<TensorId> aliasTensors;
  for (auto tId : tIds) {
    for (auto aliId : graph().allAliases(id(tId))) {
      aliasTensors.insert(idFromAliasId(aliId));
    }
  }
  return {aliasTensors.cbegin(), aliasTensors.cend()};
}

} // namespace compute
} // namespace common
} // namespace poprithms
