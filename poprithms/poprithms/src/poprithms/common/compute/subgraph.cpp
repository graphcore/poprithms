// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/rsubgraph_impl.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

template class RSubGraph<Tensor>;

SubGraph::SubGraph(SubGraphId id, Graph &g) : RSubGraph<Tensor>(id, g) {}

OpId BaseSubGraph::call(
    const SubGraphId callee,
    const std::vector<std::pair<TensorId, TensorId>> &subGraphIns,
    const TensorIds &outsInCallee) {
  auto copyIns         = CopyIns::zip(subGraphIns, CalleeIndex(0));
  auto ab              = CopyIns::split(subGraphIns);
  TensorInfos outInfos = graph().tensorInfos(outsInCallee);

  auto op = graph().template createComputeOp<Call>(
      ab.first, id(), outInfos, ab.second, callee, outsInCallee);

  registerCopies(op);
  return op;
}

void BaseSubGraph::registerCopies(OpId opId) {

  const auto *pOpWithCallees =
      graph().template dynamicCast<WithCallees>(opId);

  // For each of these copy-in destinations, register that it is the
  // destination for a CallEvent with #opId.
  auto &&inCopyDsts = pOpWithCallees->inDsts();

  for (uint64_t i = 0; i < inCopyDsts.size(); ++i) {
    auto &&tId = inCopyDsts[i].tId();
    auto &&ci  = inCopyDsts[i].calleeIndex();
    CallEvent ce(opId, pOpWithCallees->callee(ci), ci);
    graph().op(tId.opId()).insertInCopy(tId.outIndex(), ce);
  }

  // For each of these output copies, register that is is the source for a
  // CallEvent with #opId.
  auto &&outs_ = pOpWithCallees->outs();

  for (uint64_t o = 0; o < outs_.nOutTensors(); ++o) {
    for (uint64_t c = 0; c < outs_.nCallees(); ++c) {
      if (outs_.hasValue(o, c)) {
        auto &&tId = outs_.outSource(o, c);
        CallEvent ce(opId, pOpWithCallees->callee(c), c);
        graph().op(tId.opId()).insertOutCopy(tId.outIndex(), ce);
      }
    }
  }
}

} // namespace compute
} // namespace common
} // namespace poprithms
