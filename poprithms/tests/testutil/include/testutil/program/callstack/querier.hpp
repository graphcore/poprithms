// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_PROGRAM_CALLSTACK_QUERIER_HPP
#define TESTUTIL_PROGRAM_CALLSTACK_QUERIER_HPP

#include "graph.hpp"

#include <sstream>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copymap.hpp>
#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/program/callstack/querier.hpp>

namespace poprithms {
namespace program {
namespace callstack_test {

using callstack::CallStack;
using poprithms::program::callstack::CalleeTensorId;

/**
 * Completion of the callstack::Querier interface, used for running algorithms
 * on the callstack_test::Graph class.
 * */
class Querier final : public poprithms::program::callstack::Querier {
private:
  const Graph &g_;
  callstack::CopyInMap copyIns_;
  callstack::CopyOutMap copyOuts_;

public:
  Querier(const Graph &g) : g_(g) {
    copyIns_  = callstack::CopyInMap(*this);
    copyOuts_ = callstack::CopyOutMap(*this);
  }

  bool isCopyToCalleeInIndex(OpId opId, InIndex inIndex) const final {
    auto inds = g().op(opId).nonCalleeCopyInIndices();
    bool isNonCalleeCopyIn =
        (std::find(inds.cbegin(), inds.cend(), inIndex) != inds.cend());
    return !isNonCalleeCopyIn;
  }

  CalleeTensorId dstInCallee(OpId opId, InIndex inIndex) const final {
    auto i = g().op(opId).inCopies().calleeIndex(inIndex);
    auto t = g().op(opId).inCopies().dst(inIndex);
    return CalleeTensorId(t, i);
  }

  std::vector<std::pair<InIndex, TensorId>>
  copyInDsts(OpId opId) const final {
    return g().op(opId).copyInDsts();
  }

  std::vector<CopyIn> copyIns(OpId opId) const {
    return g().op(opId).inCopies().copyIns();
  }

  SubGraphId subGraphId(OpId opId) const final {
    return g().op(opId).subGraphId();
  }

  const Graph &g() const { return g_; }

  bool isCarriedTo(const TensorId &tId, const CallStack &cs) const final {
    return g().isCarriedTo(tId, cs);
  }

  TensorId carriedFrom(const TensorId &tId, const CallStack &cs) const final {
    return g().carriedFrom(tId, cs);
  }

  bool isCarriedFrom(const TensorId &tId, const CallStack &cs) const final {
    return g().isCarriedFrom(tId, cs);
  }
  TensorId carriedTo(const TensorId &tId, const CallStack &cs) const final {
    return g().carriedTo(tId, cs);
  }

  uint64_t nOutTensors(OpId i) const final { return g().nOutTensors(i); }

  SubGraphIds callees(OpId i) const final { return g().callees(i); }

  InIndices nonCalleeCopyInIndices(OpId opId) const final {
    return g().op(opId).nonCalleeCopyInIndices();
  }

  TensorIds inTensorIds(OpId opId) const final {
    return g().inTensorIds(opId);
  }

  TensorId inTensorId(OpId opId, InIndex inIndex) const final {
    return g().inTensorId(opId, inIndex);
  }

  OpIds opIds() const final { return g().opIdsAllSubGraphs(); }

  OpIds opIds(SubGraphId sg) const final { return g().opIds(sg); }

  std::string str(OpId id) const final { return std::to_string(id.get()); }

  bool isDstInCallee(const TensorId &tId, const CallEvent &cse) const {
    return g().op(cse.caller()).inCopies().isDst(cse.index(), tId);
  }

  bool isSrcInCallee(const TensorId &tId, const CallEvent &cse) const final {
    return g().op(cse.caller()).outCopies().isSource(cse.index(), tId);
  }

  const CopyIns &inCopies(OpId opId) const { return g().op(opId).inCopies(); }

  const CopyOuts &outCopies(OpId opId) const {
    return g().op(opId).outCopies();
  }

  TensorId srcInCaller(const TensorId &inCallee,
                       const CallEvent &cse) const final {
    return g().op(cse.caller()).inCopies().src(cse.index(), inCallee);
  }

  TensorId srcInCallee(const CallEvent &cse, OutIndex o) const final {
    return g().op(cse.caller()).outCopies().outSource(o, cse.index());
  }

  TensorId dstInCaller(const TensorId &inCallee,
                       const CallEvent &ce) const final {
    auto index =
        g().op(ce.caller()).outCopies().outIndex(ce.index_u64(), inCallee);
    return TensorId{ce.caller(), index};
  }

  bool hasSrcInCallee(const CallEvent &cse, OutIndex o) const final {
    return g().op(cse.caller()).outCopies().hasValue(o, cse.index());
  }

  std::vector<std::pair<CallEvent, InIndex>>
  getCopyInsTo(const TensorId &inCallee) const final {
    return copyIns_.get(inCallee);
  }

  std::vector<std::pair<CallEvent, OutIndex>>
  getCopyOutsFrom(const TensorId &inCallee) const final {
    return copyOuts_.get(inCallee);
  }

  ConsumptionIds consumptionIds(const TensorId &tId) const final {
    return g().consumptionIds(tId);
  }

  bool hasConsumers(const TensorId &tId) const {
    return g().hasConsumptionIds(tId);
  }
};

} // namespace callstack_test
} // namespace program
} // namespace poprithms

#endif
