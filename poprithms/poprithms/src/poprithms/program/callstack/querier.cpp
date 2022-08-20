// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <set>
#include <sstream>

#include <poprithms/common/multiout/fwdedgemap.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/program/callstack/querier.hpp>
#include <poprithms/program/callstack/stackutil.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/schedule/vanilla/vanillamap.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;

namespace {

class SingleGraphBack {
public:
  SingleGraphBack(const Querier &gq) : gq_(gq) {}
  TensorIds neighbors(const TensorId &n) const {
    return gq_.inTensorIds(n.opId());
  }

private:
  const Querier &gq_;
};

class MultiGraphBack {
public:
  MultiGraphBack(const Querier &gq_) : gq(gq_) {}
  StackTensorIds neighbors(const StackTensorId &n) const;

private:
  const Querier &gq;
};

class SingleGraphForward {
public:
  SingleGraphForward(const Querier &gq) : gq_(gq) {}
  TensorIds neighbors(const TensorId &n) const {
    return gq_.outTensorIds(n.opId());
  }

private:
  const Querier &gq_;
};

class MultiGraphForward {
public:
  MultiGraphForward(const Querier &gq_) : gq(gq_) {}
  StackTensorIds neighbors(const StackTensorId &n) const;

private:
  const Querier &gq;
};

StackTensorIds MultiGraphBack::neighbors(const StackTensorId &source0) const {

  const auto tId        = source0.tId();
  const auto sourceOpId = tId.opId();
  auto callStack        = source0.callStack();
  StackTensorIds toReturn;

  auto callees_ = gq.callees(tId.opId());

  // Traverse back into the callees of the op (if it has any callees).
  // Specifically, for every output in the caller, traverse back to the source
  // of the copy in the callee.
  for (uint64_t i = 0; i < callees_.size(); ++i) {

    // Going into callee, so stack gets larger by 1:
    auto callStack_ = callStack;
    const CallEvent cse{tId.opId(), callees_[i], i};
    if (gq.hasSrcInCallee(cse, tId.outIndex())) {
      callStack_.push_back(cse);
      toReturn.push_back(StackTensorId(gq.srcInCallee(cse, tId.outIndex()),
                                       std::move(callStack_)));
    }
  }

  for (InIndex i : gq.nonCalleeCopyInIndices(sourceOpId)) {
    toReturn.push_back({gq.inTensorId(sourceOpId, i), callStack});
  }

  if (gq.isCarriedTo(tId, callStack)) {
    toReturn.push_back({gq.carriedFrom(tId, callStack), callStack});
  }

  // If the tensor is the destination of a copy into a callee, traverse to the
  // source of the copy.
  if (!callStack.empty() && gq.isDstInCallee(tId, callStack.back())) {

    const auto currentCse = callStack.back();
    callStack.pop_back();
    toReturn.push_back(
        StackTensorId(gq.srcInCaller(tId, currentCse), std::move(callStack)));
  }

  return toReturn;
}

StackTensorIds
MultiGraphForward::neighbors(const StackTensorId &source0) const {

  const auto tId = source0.tId();
  auto callStack = source0.callStack();
  StackTensorIds toReturn;

  for (auto consumptionId : gq.consumptionIds(tId)) {
    auto consumingOp          = consumptionId.opId();
    auto inIndexOfConsumption = consumptionId.inIndex();

    auto callees = gq.callees(consumingOp);

    //   (1) if the tensor is consumed by an op with a callee, and the the
    //       tensor is copied into the callee sub-graph, traverse to the
    //       destination of the copy.
    if (gq.isCopyToCalleeInIndex(consumingOp, inIndexOfConsumption)) {

      auto dstInCallee = gq.dstInCallee(consumingOp, inIndexOfConsumption);
      auto ci          = dstInCallee.calleeIndex();
      auto callee      = callees.at(ci.get());

      auto callStack_ = callStack;
      callStack_.push_back(CallEvent{consumingOp, callee, ci});
      StackTensorId sti(dstInCallee.tId(), std::move(callStack_));
      toReturn.push_back(sti);
    }

    //   (2) if the tensor is consumed by an op and is not copied to a callee
    //       sub-graph in the consumer, traverse to all of the op's outputs.
    else {
      for (auto o : gq.outTensorIds(consumingOp)) {
        toReturn.push_back({o, callStack});
      }
    }
  }

  if (gq.isCarriedFrom(tId, callStack)) {
    toReturn.push_back({gq.carriedTo(tId, callStack), callStack});
  }

  //    (3) if the tensor is in a callee sub-graph and is copied out, traverse
  //        to the destination of the copy. In this case, the stack size
  //        decreases by 1 for the destination.
  if (!callStack.empty() && gq.isSrcInCallee(tId, callStack.back())) {
    const auto currentCse = callStack.back();
    callStack.pop_back();
    toReturn.push_back(
        StackTensorId(gq.dstInCaller(tId, currentCse), std::move(callStack)));
  }

  return toReturn;
}

} // namespace

TensorIds Querier::onSingleGraphPathTo(const TensorIds &tIds) const {

  SingleGraphBack bb(*this);
  TensorIds out = poprithms::common::multiout::depthFirst(
      bb, tIds, [](const TensorId &) { return true; });
  return out;
}

TensorIds Querier::onSingleGraphPathFrom(const TensorIds &tIds) const {

  SingleGraphForward bb(*this);
  TensorIds out = poprithms::common::multiout::depthFirst(
      bb, tIds, [](const TensorId &) { return true; });
  return out;
}

StackTensorIds Querier::onMultiGraphPathTo(
    const StackTensorIds &tIds,
    const std::function<bool(StackTensorId)> &accept) const {
  return poprithms::common::multiout::depthFirst(
      MultiGraphBack(*this), tIds, accept);
}

StackTensorIds Querier::onMultiGraphPathFrom(
    const StackTensorIds &tIds,
    const std::function<bool(StackTensorId)> &accept) const {
  return poprithms::common::multiout::depthFirst(
      MultiGraphForward(*this), tIds, accept);
}

StackTensorIds Querier::nestedFullStack(const SubGraphIds &stackBase) const {

  StackTensorIds stackTensorIds;
  std::vector<std::pair<CallStack, SubGraphId>> toProcess;
  for (auto sg0 : stackBase) {
    toProcess.push_back({{}, sg0});
  }

  while (!toProcess.empty()) {
    auto current         = toProcess.back();
    auto currentStack    = current.first;
    auto currentSubGraph = current.second;
    toProcess.pop_back();

    for (auto opId : opIds(currentSubGraph)) {
      for (auto tId : outTensorIds(opId)) {
        stackTensorIds.push_back({tId, currentStack});
      }

      auto callees_ = callees(opId);
      for (uint64_t calleeIndex = 0; calleeIndex < callees_.size();
           ++calleeIndex) {
        const auto nxtSubGraph = callees_[calleeIndex];
        auto nxtStack          = currentStack;
        nxtStack.push_back({opId, nxtSubGraph, calleeIndex});
        toProcess.push_back({nxtStack, nxtSubGraph});
      }
    }
  }
  return stackTensorIds;
}

std::map<TensorId, std::vector<CallStack>>
Querier::nestedFullStackMap(const SubGraphIds &stackBase) const {
  const auto fwdAll = nestedFullStack(stackBase);
  std::map<TensorId, std::vector<CallStack>> tmap;
  for (const auto &x : fwdAll) {
    const auto tId = x.tId();
    auto found     = tmap.find(tId);
    if (found == tmap.cend()) {
      tmap.insert({tId, {x.callStack()}});
    } else {
      found->second.push_back(x.callStack());
    }
  }
  return tmap;
}

TensorIds Querier::outTensorIds(OpId id) const {
  TensorIds ids;
  ids.reserve(nOutTensors(id));
  for (OutIndex o = 0; o < nOutTensors(id); ++o) {
    ids.push_back({id, o});
  }
  return ids;
}

OpIds Querier::scheduled(DataDepOrder ddo, GraphDepOrder gdo) const {

  poprithms::common::multiout::FwdEdgeMap opFwdMap(opIds());
  for (auto opId : opIds()) {
    for (auto t : inTensorIds(opId)) {
      opFwdMap.insertEdge(t.opId(), opId);
    }
  }
  using namespace poprithms::schedule::vanilla;
  auto opOrder = opFwdMap.unpacked(getSchedule_u64(
      opFwdMap.fwdEdgesCompact(), ErrorIfCycle::Yes, VerifyEdges::Yes));

  if (ddo == DataDepOrder::Bwd) {
    std::reverse(opOrder.begin(), opOrder.end());
  }

  auto sgOrder = topDown();
  if (gdo == GraphDepOrder::BottomUp) {
    std::reverse(sgOrder.begin(), sgOrder.end());
  }

  return stableSortBySubGraphOrder(opOrder, sgOrder);
}

SubGraphIds Querier::allSubGraphIds() const {
  std::set<SubGraphId> sgIds;
  for (auto op : opIds()) {
    sgIds.insert(subGraphId(op));
    for (auto callee : callees(op)) {
      sgIds.insert(callee);
    }
  }
  return SubGraphIds(sgIds.cbegin(), sgIds.cend());
}

SubGraphIds Querier::topDown() const {

  std::unordered_map<SubGraphId, SubGraphIds> fwd;
  for (auto op : opIds()) {
    if (fwd.count(subGraphId(op)) == 0) {
      fwd.insert({subGraphId(op), {}});
    }
    auto found = fwd.find(subGraphId(op));
    for (auto c : callees(op)) {
      found->second.push_back(c);
    }
  }

  using namespace poprithms::schedule::vanilla;
  return getSchedule(fwd, ErrorIfCycle::Yes, VerifyEdges::Yes);
}

OpIds Querier::stableSortBySubGraphOrder(const OpIds &opOrder,
                                         const SubGraphIds &sgOrder) const {

  std::unordered_map<SubGraphId, OpIds> scatter;
  for (auto sg : sgOrder) {
    scatter.insert({sg, {}});
  }
  for (auto op : opOrder) {
    scatter[subGraphId(op)].push_back(op);
  }

  OpIds gather;
  gather.reserve(opOrder.size());
  for (auto sg : sgOrder) {
    for (auto op : scatter.at(sg)) {
      gather.push_back(op);
    }
  }
  return gather;
}

void Querier::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

} // namespace callstack
} // namespace program
} // namespace poprithms
