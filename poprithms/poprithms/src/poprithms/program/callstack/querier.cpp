// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <limits>
#include <set>
#include <sstream>

#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/program/callstack/querier.hpp>
#include <poprithms/program/callstack/stackutil.hpp>

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

StackTensorIds MultiGraphBack::neighbors(const StackTensorId &source0) const {

  const auto tId = source0.tId();
  auto callStack = source0.callStack();
  StackTensorIds toReturn;

  auto callees_ = gq.callees(tId.opId());

  // If creator has callees, traverse back into the callees.
  if (!callees_.empty()) {
    for (uint64_t i = 0; i < callees_.size(); ++i) {
      auto callStack_ = callStack;
      const CallEvent cse{tId.opId(), callees_[i], i};
      callStack_.push_back(cse);
      toReturn.push_back(StackTensorId(gq.srcInCallee(cse, tId.outIndex()),
                                       std::move(callStack_)));
    }
  }

  // If creator has no callees, just traverse to creator's inputs.
  else {
    auto toAppend =
        callstack::StackUtil::inScope(gq.inTensorIds(tId.opId()), callStack);
    toReturn.insert(toReturn.end(), toAppend.cbegin(), toAppend.cend());
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

} // namespace

TensorIds Querier::onSingleGraphPathTo(const TensorIds &tIds) const {

  SingleGraphBack bb(*this);
  TensorIds out = poprithms::common::multiout::depthFirst(
      bb, tIds, [](const TensorId &) { return true; });
  return out;
}

StackTensorIds Querier::onMultiGraphPathTo(const StackTensorIds &tIds) const {
  auto stids = poprithms::common::multiout::depthFirst(
      MultiGraphBack(*this), tIds, [](const StackTensorId &) {
        return true;
      });
  return stids;
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

} // namespace callstack
} // namespace program
} // namespace poprithms
