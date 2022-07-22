// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <unordered_set>

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
  auto &&outputs = pOpWithCallees->outs();

  for (uint64_t o = 0; o < outputs.nOutTensors(); ++o) {
    for (uint64_t c = 0; c < outputs.nCallees(); ++c) {
      if (outputs.hasValue(o, c)) {
        auto &&tId = outputs.outSource(o, c);
        CallEvent ce(opId, pOpWithCallees->callee(c), c);
        graph().op(tId.opId()).insertOutCopy(tId.outIndex(), ce);
      }
    }
  }
}

TensorIds BaseSubGraph::tensorIds() const { return graph().tensorIds(id()); }

TensorIds BaseSubGraph::tensorIds(DeviceType dt) const {
  auto all = tensorIds();
  TensorIds filtered;
  for (const auto &x : all) {
    if (graph().deviceType(x) == dt) {
      filtered.push_back(x);
    }
  }
  return filtered;
}

OpIds BaseSubGraph::constInitIds() const {
  return graph().opIds<poprithms::common::compute::ConstInit>(id());
}
OpIds BaseSubGraph::varInitIds() const {
  return graph().opIds<poprithms::common::compute::VarInit>(id());
}
OpIds BaseSubGraph::initIds() const {
  return graph().opIds<poprithms::common::compute::Init>(id());
}

OpId BaseSubGraph::repeat(
    SubGraphId callee,
    uint64_t repeatCount,
    const std::vector<std::pair<TensorId, TensorId>> &stackedInputs,
    const CarriedTensorIds &carriedInputs,
    const std::vector<std::pair<TensorId, IsStackedCopy>> &outputs,
    StackedCopyOrder stackedCopyOrder) {

  const auto nInputs = stackedInputs.size() + carriedInputs.nTensors();

  // All input copy sources:
  TensorIds copyInSources;
  copyInSources.reserve(nInputs);

  // All input copy destinations:
  TensorIds copyInDestinations;
  copyInDestinations.reserve(nInputs);

  // Set the inputs:
  {
    for (auto &&carriedTensorId : carriedInputs.carriedTensorIds()) {
      copyInSources.push_back(carriedTensorId.sourceInCaller());
      copyInDestinations.push_back(carriedTensorId.destinationInCallee());
    }

    for (auto rt : stackedInputs) {
      copyInSources.push_back(rt.first);
      copyInDestinations.push_back(rt.second);
    }
  }

  // The source and destination in the callee of the carries between
  // iterations:
  TensorIds carryFrom;
  carryFrom.reserve(carriedInputs.nTensors());

  TensorIds carryTo;
  carryTo.reserve(carriedInputs.nTensors());

  // Set the carry sources and destinations:
  {
    for (auto p : carriedInputs.carriedTensorIds()) {
      carryFrom.push_back(p.sourceInCallee());
      carryTo.push_back(p.destinationInCallee());
    }
  }

  // The sources of the output copies, in the callees:
  TensorIds copyOutSources;
  copyOutSources.reserve(outputs.size());

  // The information about the the outputs, in this sub-graph (i.e. the
  // outputs in the caller's sub-graph).
  std::vector<TensorInfo> outInfos;
  outInfos.reserve(outputs.size());

  // Set the output vectors required for the Repeat op constructor:
  {
    for (auto rt : outputs) {
      auto isStacked = rt.second == IsStackedCopy::Yes;
      auto &&outInfo = graph().tensorInfo(rt.first);
      if (isStacked) {
        outInfo = outInfo.withShape(outInfo.shape().prepend(repeatCount));
      }
      copyOutSources.push_back(rt.first);
      outInfos.push_back(outInfo);
    }
  }

  auto opId = graph().createComputeOp<Repeat>(copyInSources,
                                              id(),
                                              outInfos,
                                              callee,
                                              repeatCount,
                                              copyInDestinations,
                                              copyOutSources,
                                              carryFrom,
                                              carryTo,
                                              stackedCopyOrder);

  registerCopies(opId);

  return opId;
}

OpId BaseSubGraph::repeatAllOut(
    SubGraphId callee,
    uint64_t repeatCount,
    const std::vector<std::pair<TensorId, TensorId>> &stackedInputs,
    const CarriedTensorIds &carriedInputs,
    const TensorIds &flatOutputs,
    StackedCopyOrder stackedCopyOrder) {

  auto allCalleeTensors = graph().tensorIds(callee);

  // All tensors in #callee which are NOT flat outputs:
  auto nonFlats = poprithms::common::multiout::Graph::setDifference(
      allCalleeTensors, flatOutputs);

  std::vector<std::pair<TensorId, IsStackedCopy>> outputs;
  outputs.reserve(allCalleeTensors.size());

  for (auto t : flatOutputs) {
    outputs.push_back({t, IsStackedCopy::No});
  }

  // For all the tensors which are not forced to be flat by being in
  // #flatOutputs, we try to make them stacked outputs. Tensors which are
  // carry sources cannot be stacked (this is a constraint imposed by the
  // design of the repeat op).
  std::unordered_set<TensorId> mustBeFlat;
  for (auto &&c : carriedInputs.carriedTensorIds()) {
    mustBeFlat.insert(c.sourceInCallee());
  }

  for (auto &&otId : nonFlats) {
    outputs.push_back({otId,
                       mustBeFlat.count(otId) == 0 ? IsStackedCopy::Yes
                                                   : IsStackedCopy::No});
  }

  return repeat(callee,
                repeatCount,
                stackedInputs,
                carriedInputs,
                outputs,
                stackedCopyOrder);
}

} // namespace compute
} // namespace common
} // namespace poprithms
