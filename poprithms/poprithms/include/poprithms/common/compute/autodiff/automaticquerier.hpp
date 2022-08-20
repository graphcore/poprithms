// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTOMATICQUERIER_HPP
#define POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTOMATICQUERIER_HPP

#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>

using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OpTraversal;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::ndarray::Shape;
using poprithms::program::callstack::CalleeIndex;
using poprithms::program::callstack::CalleeTensorId;
using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;

namespace poprithms {
namespace common {
namespace compute {

/**
 * Completion of the IAutomaticQuerier interface for a compute::Graph.
 * */
class AutomaticQuerier final
    : public poprithms::autodiff::automatic::IAutomaticQuerier {

private:
  // a constant reference to graph which will be querier:
  const Graph &graph_;

  // 2 shortcut methods to access ops in the graph:
  const Op &op(OpId id) const { return graph_.computeOp(id); }
  const WithCallees &withCallees(OpId opId) const {
    return *graph_.castOrThrow<WithCallees>(opId);
  }

public:
  AutomaticQuerier(const Graph &m_) : graph_(m_) {}
  ~AutomaticQuerier() override = default;

  ConsumptionIds consumptionIds(const TensorId &tId) const final {
    return graph_.consumptionIds(tId);
  }

  TensorId inTensorId(OpId opId, InIndex i) const final {
    return graph_.inTensorId(opId, i);
  }

  bool gradientPropagates(OpId opId, OutIndex o, InIndex i) const final {
    return graph_.gradientPropagates(OpTraversal(i, opId, o));
  }

  TensorId
  outSource(OpId opId, OutIndex outIndex, CalleeIndex ci) const final {
    return withCallees(opId).outs().outSource(outIndex, ci);
  }

  bool
  isOutSource(OpId opId, CalleeIndex ci, const TensorId &tId) const final {
    return withCallees(opId).outs().isSource(ci, tId);
  }

  OutIndex
  copyOutIndex(OpId opId, CalleeIndex ci, const TensorId &tId) const final {
    return withCallees(opId).outs().outIndex(ci, tId);
  }

  SubGraphId subGraphId(OpId opId) const final {
    return graph_.subGraphId(opId);
  }

  uint64_t nOutTensors(OpId opId) const final {
    return graph_.nOutTensors(opId);
  }

  poprithms::autodiff::guide::Objective
  localObjective(OpId opId,
                 CalleeIndex ci,
                 const InIndices &fromTargets,
                 const OutIndices &gradsIn) const final {
    return withCallees(opId).localObjective(ci, fromTargets, gradsIn);
  }

  std::string str(OpId opId) const final { return op(opId).str(); }

  uint64_t nCallees(OpId opId) const final { return graph_.nCallees(opId); }

  uint64_t nInTensors(OpId opId) const final {
    return graph_.nInTensors(opId);
  }

  uint64_t nInCopies(OpId opId) const final {
    return op(opId).nInputsCopiedToCallees();
  }

  CalleeTensorId inDst(OpId opId, InIndex i) const final {
    return op(opId).dstInCallee(i);
  }

  SubGraphId callee(OpId opId, CalleeIndex ci) const final {
    return op(opId).callee(ci);
  }

  CalleeIndex inDstCalleeIndex(OpId opId, InIndex inIndex) const final {
    return op(opId).dstInCallee(inIndex).calleeIndex();
  }

  SubGraphId subGraphId(const TensorId &tId) const final {
    return graph_.subGraphId(tId);
  }

  CallEvent event(OpId opId, CalleeIndex ci) const final {
    return CallEvent(opId, op(opId).callee(ci), ci);
  }

  bool isDefinitelyAllConstZero(const TensorId &tId) const final;

  Shape shape(const TensorId &tId) const final { return graph_.shape(tId); }

  TensorIds tensorIds(SubGraphId sgId) const final {
    return graph_.tensorIds(sgId);
  }

  TensorId dstInCaller(const TensorId &tId, const CallEvent &ce) const final {
    return graph_.dstInCaller(tId, ce);
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
