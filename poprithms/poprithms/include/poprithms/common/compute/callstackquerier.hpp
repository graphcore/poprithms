// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_PRUNE_CALLSTACKQUERIER_HPP
#define POPRITHMS_COMMON_COMPUTE_PRUNE_CALLSTACKQUERIER_HPP

#include <memory>
#include <vector>

#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/program/callstack/querier.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::program::callstack::CallStack;

/**
 * Completion of the interface callstack::Querier.
 * */
class CallstackQuerier final : public poprithms::program::callstack::Querier {

private:
  const Graph &graph_;
  const Graph &graph() const { return graph_; }
  const WithCallees &wc(OpId opId) const {
    return *graph().castOrThrow<WithCallees>(opId);
  }

public:
  CallstackQuerier(const Graph &m)
      : poprithms::program::callstack::Querier(), graph_(m) {}

  std::vector<std::pair<InIndex, TensorId>> copyInDsts(OpId opId) const final;

  std::vector<CopyIn> copyIns(OpId opId) const {
    return graph().castOrThrow<WithCallees>(opId)->copyIns();
  }

  SubGraphId subGraphId(OpId opId) const { return graph().subGraphId(opId); }

  uint64_t nOutTensors(OpId id) const final {
    return graph().nOutTensors(id);
  }

  bool isCopyToCalleeInIndex(OpId opId, InIndex inIndex) const final {
    return graph().computeOp(opId).isCopyToCalleeInIndex(inIndex);
  }

  CalleeTensorId dstInCallee(OpId opId, InIndex inIndex) const final {
    return graph().computeOp(opId).dstInCallee(inIndex);
  }

  SubGraphIds callees(OpId id) const final { return graph().callees(id); }

  TensorIds inTensorIds(OpId id) const final {
    return graph().inTensorIds(id);
  }

  TensorId inTensorId(OpId id, InIndex inIndex) const final {
    return graph().inTensorId(id, inIndex);
  }

  InIndices nonCalleeCopyInIndices(OpId opId) const final;

  OpIds opIds() const final { return graph().opIds(); }

  OpIds opIds(SubGraphId sg) const final { return graph().opIds(sg); }

  std::string str(OpId id) const final { return graph().str(id); }

  bool isDstInCallee(const TensorId &tId, const CallEvent &cse) const final {
    return graph().isDstInCallee(tId, cse);
  }

  CopyOuts outCopies(OpId opId) const {
    return graph().castOrThrow<WithCallees>(opId)->outs();
  }

  TensorId dstInCaller(const TensorId &inCallee,
                       const CallEvent &ce) const final {
    return graph().dstInCaller(inCallee, ce);
  }

  bool isSrcInCallee(const TensorId &tId, const CallEvent &ce) const final {
    return graph().computeOp(tId.opId()).isSrcInCallee(tId.outIndex(), ce);
  }

  bool isCarriedTo(const TensorId &, const CallStack &) const final;
  TensorId carriedFrom(const TensorId &, const CallStack &) const final;

  bool isCarriedFrom(const TensorId &, const CallStack &) const final;
  TensorId carriedTo(const TensorId &, const CallStack &) const final;

  TensorId srcInCaller(const TensorId &inCallee,
                       const CallEvent &cse) const final {
    return graph().srcInCaller(inCallee, cse);
  }

  TensorId srcInCallee(const CallEvent &cse, OutIndex o) const final {
    return graph().srcInCallee(cse, o);
  }

  bool hasSrcInCallee(const CallEvent &cse, OutIndex o) const final {
    return graph().hasSrcInCallee(cse, o);
  }

  bool hasConsumers(const TensorId &tId) const final {
    return graph().hasConsumptionIds(tId);
  }

  std::vector<std::pair<CallEvent, InIndex>>
  getCopyInsTo(const TensorId &inCallee) const final {
    return graph().indexedInCopies(inCallee);
  }

  std::vector<std::pair<CallEvent, OutIndex>>
  getCopyOutsFrom(const TensorId &inCallee) const final {
    return graph().indexedOutCopies(inCallee);
  }

  ConsumptionIds consumptionIds(const TensorId &tId) const final {
    return graph().consumptionIds(tId);
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
