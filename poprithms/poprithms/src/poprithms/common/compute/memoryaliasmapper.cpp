// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/memoryaliasmapper.hpp>
#include <poprithms/compute/host/regionutil.hpp>
#include <poprithms/memory/alias/jitgrower.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::InIndex;

/**
 * Just-in-time helper class for adding (growing) only the tensors necessary,
 * when necessary.
 * */
class JitAliasGrower : public poprithms::memory::alias::JitGrower<TensorId> {
private:
  const Graph &graph_;
  MemoryAliasMapper &memoryAliasMapper_;

public:
  JitAliasGrower(const Graph &g, MemoryAliasMapper &mam)
      : graph_(g), memoryAliasMapper_(mam) {}

  /**
   * All aliasing inputs of the op which creates #nxt, as well as the root
   * reference of #nxt.
   * */
  TensorIds aliasingIns(const TensorId &nxt) const final {
    const auto &op_ = graph_.computeOp(nxt.opId());
    TensorIds aliIns;
    for (InIndex i = 0; i < op_.nInTensors(); ++i) {
      if (op_.aliases(i, nxt.outIndex())) {
        aliIns.push_back(op_.inTensorId(i));
      }
    }
    const auto root = op_.rootRef(nxt.outIndex());
    if (root != nxt) {
      aliIns.push_back(root);
    }
    return aliIns;
  }

  void growAliasTensors(const TensorIds &sched) final {
    for (auto &&tId : sched) {
      if (!memoryAliasMapper_.has(tId)) {
        graph_.computeOp(tId.opId()).growAliasMapper(memoryAliasMapper_);
      }
    }
  }

  bool containsAliasTensor(const TensorId &tId) const final {
    return memoryAliasMapper_.has(tId);
  }
};

void MemoryAliasMapper::extend(const TensorIds &tIds) {
  JitAliasGrower jag(graph_, *this);
  jag.extend(tIds);
}

MemoryAliasMapper::MemoryAliasMapper(const Graph &g, const TensorIds &tIds)
    : poprithms::memory::alias::Mapper<TensorId>(), graph_(g) {
  extend(tIds);
}

bool AliasQuerier::isAllConstZero(const Graph &g, const TensorId &tId) {

  MemoryAliasMapper mam(g, {tId});

  // If any of the allocations of #tId are non-constant, return false.
  if (mam.graph().containsColor(mam.id(tId), MemoryAliasVariable)) {
    return false;
  }
  auto aliases = mam.idsFromAliasIds(mam.graph().allAliases(mam.id(tId)));

  OpIds allocs;

  // For all the aliases of #tId which are constant initializers, check that
  // the elements of the regions used by #tId are all 0.
  for (auto a : aliases) {
    if (g.isConstInit(a.opId())) {
      auto &&regs = mam.graph().allocRegions(mam.id(tId), mam.id(a));

      for (const auto &regs_ : regs) {
        for (const auto &r : regs_.get()) {
          if (!poprithms::compute::host::RegionUtil::allZero(
                  g.constInitValue(a.opId()), r)) {
            return false;
          }
        }
      }

      allocs.push_back(a.opId());
    }
  }

  return true;
}

} // namespace compute
} // namespace common
} // namespace poprithms
