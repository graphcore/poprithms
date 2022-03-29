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
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copymap.hpp>
#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/program/callstack/querier.hpp>

namespace poprithms {
namespace program {
namespace callstack_test {

/**
 * Completion of the callstack::Querier interface, used for running algorithms
 * on the callstack_test::Graph class.
 * */
class Querier : public poprithms::program::callstack::Querier {
private:
  const Graph &g_;
  callstack::CopyInMap copyIns_;
  callstack::CopyOutMap copyOuts_;

public:
  Querier(const Graph &g) : g_(g) {
    copyIns_  = callstack::CopyInMap(*this);
    copyOuts_ = callstack::CopyOutMap(*this);
  }

  const Graph &g() const { return g_; }

  uint64_t nOutTensors(OpId i) const final { return g().nOutTensors(i); }

  uint64_t nInTensors(OpId i) const final { return g().nInTensors(i); }

  SubGraphIds callees(OpId i) const final { return g().callees(i); }

  TensorIds inTensorIds(OpId i) const final { return g().inTensorIds(i); }

  OpIds opIds() const final { return g().opIdsAllSubGraphs(); }

  OpIds opIds(SubGraphId sg) const final { return g().opIds(sg); }

  std::string str(OpId id) const final { return std::to_string(id.get()); }

  bool isDstInCallee(const TensorId &tId, const CallEvent &cse) const {
    return g().op(cse.caller()).inCopies().isDst(cse.index(), tId);
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

  ConsumptionIds consumptionIds(const TensorId &tId) const final {
    return g().consumptionIds(tId);
  }

  std::vector<std::pair<CallEvent, InIndex>>
  getCopyInsTo(const TensorId &inCallee) const final {
    return copyIns_.get(inCallee);
  }

  std::vector<std::pair<CallEvent, OutIndex>>
  getCopyOutsFrom(const TensorId &inCallee) const final {
    return copyOuts_.get(inCallee);
  }
};

} // namespace callstack_test
} // namespace program
} // namespace poprithms

#endif
