// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/memoryaliasmapper.hpp>
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

} // namespace compute
} // namespace common
} // namespace poprithms
