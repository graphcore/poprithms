// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_AUTODIFF_GUIDEGRAPHINFO_HPP
#define POPRITHMS_COMMON_COMPUTE_AUTODIFF_GUIDEGRAPHINFO_HPP

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/automatic/requiredids.hpp>
#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/ids/ids.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/scheduler.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::compute::Graph;

/**
 * Implementation of the guide::GraphInfo interface for a compute::Graph.
 * */
class GuideGraphInfo : public autodiff::guide::GraphInfo {
public:
  virtual ~GuideGraphInfo() = default;

  GuideGraphInfo(const Graph &g, const autodiff::automatic::GradInfos &gis_)
      : graph_(g), gradInfos(gis_) {}
  GuideGraphInfo() = delete;

  bool gradientPropagates(const OpTraversal &o) const final {
    return graph_.gradientPropagates(o);
  }

  OpIds subSchedule(const std::set<OpId> &opIds) const final {
    return graph_.vanillaSubSchedule(opIds);
  }

  virtual void appendOpInfo(std::ostream &ost, OpId id) const final {
    graph_.appendOpColumns(ost, {id});
  }

  void extendAutodiffRequiredTensors(OpId id,
                                     std::set<TensorId> &s) const final;

  TensorIds inTensorIds(OpId id) const final {
    return graph_.inTensorIds(id);
  }

  TensorId inTensorId(OpId id, InIndex ind) const final {
    return graph_.inTensorId(id, ind);
  }

  uint64_t nInTensors(OpId id) const final { return graph_.nInTensors(id); }

  uint64_t nOutTensors(OpId id) const final { return graph_.nOutTensors(id); }

  ConsumptionIds consumptionIds(const TensorId &id) const final {
    return graph_.consumptionIds(id);
  }

  /// All targets and tensors with provided gradients must be in the same
  /// sub-graph.
  void assertValidPaths(const TensorIds &targets,
                        const TensorIds &gradsProvidedFor) const final;

  /// VarInits cannot be rerun if their output value is required.
  void assertCanBeRerun(OpId opId, bool valueRequired) const final;

  /// Fixed point tensors cannot have gradients.
  void assertCanHaveGrad(const TensorId &tId) const final;

  bool isValueDependent(const OpTraversal &ot) const final {
    return graph_.computeOp(ot.opId()).isValueDependent(ot.inIndex(),
                                                        ot.outIndex());
  }

private:
  const Graph &graph_;
  const autodiff::automatic::GradInfos &gradInfos;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
