// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <autodiff/autodiff/error.hpp>
#include <map>

#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/common/multiout/tensormap.hpp>
#include <poprithms/util/map.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/where.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

OptionalTensorIds Autodiff::optionalNonGrads(const TensorIds &ids) const {
  return poprithms::util::whereIdsInMap<OptionalTensorId>(nonGrads, ids);
}

OptionalTensorIds Autodiff::optionalGrads(const TensorIds &ids) const {
  return poprithms::util::whereIdsInMap<OptionalTensorId>(grads, ids);
}

/**
 * All of the non-gradient tensors in the backwards graph. This is a map
 * from TensorIds in the forward graph, to non-gradient tensors in the
 * backward graph. These tensors are either checkpoints or recomputed.
 * */
void Autodiff::setNonGrads() {
  using namespace poprithms::common;
  for (auto c : objective.checkpoints()) {
    auto cp = [this, c]() {
      if (!objective.isInGraph()) {
        const auto var = graphMutator.createVariable(c);
        graphMutator.setName(var.opId(), genCheckpointName(c));
        return var;
      }
      return c;
    }();
    nonGrads.insert({c, cp});
  }

  for (auto opId : guide.opsToRerun()) {
    const auto cloneOp =
        graphMutator.clone(opId,
                           poprithms::util::getValues<TensorIds, TensorId>(
                               graphInfo.inTensorIds(opId), nonGrads));
    graphMutator.setName(cloneOp, genRerunName(opId));
    for (uint64_t o = 0; o < graphInfo.nOutTensors(opId); ++o) {
      nonGrads.insert({{opId, o}, {cloneOp, o}});
    }
  }
}

/**
 * Initialize the gradient Tensors. The object 'gradients' maps from
 * TensorIds in the forward graph, to gradient tensors in the backward
 * graph. The tensors will be created and updated as autodiff is run
 * (below).
 * */
void Autodiff::initGrads() {
  for (const auto &tId : guide.nonGradsWithGrads()) {
    auto zeroed = graphMutator.createZero(tId);
    graphMutator.setName(zeroed.opId(), genInitGradName(tId));
    grads.insert({tId, zeroed});
  }
}

TensorId Autodiff::getGrad(const TensorId &nonGrad) const {
  const auto found = grads.find(nonGrad);
  if (found == grads.cend()) {
    throw error("No grad for " + nonGrad.str() + " found, in getGrad. ");
  }
  return found->second;
}

TensorId &Autodiff::getGradIdRef(const TensorId &nonGrad) {
  const auto found = grads.find(nonGrad);
  if (found == grads.cend()) {
    throw error("No grad for " + nonGrad.str() + " found, in getGradIdRef. ");
  }
  return found->second;
}

TensorId Autodiff::getNonGrad(const TensorId &nonGrad) const {
  const auto found = nonGrads.find(nonGrad);
  if (found == nonGrads.cend()) {
    throw error("No non-grad for " + nonGrad.str() + " found. ");
  }
  return found->second;
}

/**
 * The gradient input tensors for the gradient graph. This object is a map
 * from tensors in the forward graph for which gradients will be provided,
 * to said gradients in the backwards graph.
 * */
void Autodiff::setGradsIn() {

  const auto getGrad = [this](uint64_t i) {
    if (objective.isInGraph()) {
      return objective.gradsProvided()[i];
    }
    const auto nonGrad = objective.gradsProvidedFor()[i];
    const auto var     = graphMutator.createVariable(nonGrad);
    graphMutator.setName(var.opId(), genInGradName(nonGrad));
    return var;
  };

  for (uint64_t i = 0; i < objective.nInGrads(); ++i) {
    const auto id = objective.gradsProvidedFor()[i];
    gradsIn.insert({id, getGrad(i)});
  }
}

// Add input gradients to gradients.
void Autodiff::addGradsInToGrads() {
  for (const auto &[gradProvidedFor, gradIn] : gradsIn) {
    auto &t = getGradIdRef(gradProvidedFor);
    t       = graphMutator.add(t, gradIn);
  }
}

void Autodiff::backpropagate() {

  const auto fwdEdges = guide.fwdEdges();

  /**
   * This is a map from ops, to the total number of traversals they are
   * waiting for before they can back-propagate. It is the sum over all output
   * tensors of the number of traversals they are waiting for.
   * */
  auto nAwaiting = guide.getFwdEdgeDependencyCount();

  /**
   * The ops which are ready to be auto-diffed.
   * */
  OpIds ready;
  for (const auto &[opId, cnt] : nAwaiting) {
    if (cnt == 0) {
      ready.push_back(opId);
    }
  }

  while (!ready.empty()) {
    auto nxt = ready.back();
    ready.pop_back();

    // To populate: the gradients of each of the inputs of 'nxt':
    const OptionalTensorIds inGrads = graphMutator.getInGrads(nxt, *this);

    std::map<OpId, std::vector<InIndex>> creates;

    for (uint64_t i = 0; i < graphInfo.nInTensors(nxt); ++i) {
      if (inGrads[i].has_value()) {
        creates[inGrads[i].value().opId()].push_back(i);
        const auto inId = graphInfo.inTensorId(nxt, InIndex(i));
        if (grads.count(inId) != 0) {
          auto &t = grads.at(inId);
          t       = graphMutator.add(t, inGrads[i].value());
        }
      }
    }

    for (const auto &[opId, indices] : creates) {
      graphMutator.setName(opId, genGradInsName(nxt, indices));
    }
    for (auto opId_ : fwdEdges.at(nxt)) {
      --nAwaiting[opId_];
      if (nAwaiting[opId_] == 0) {
        ready.push_back(opId_);
      }
    }
  }
  summary_.setGradsIn(util::getValues<TensorIds, TensorId>(
      objective.gradsProvidedFor(), gradsIn));
  summary_.setCheckpointsIn(util::getValues<TensorIds, TensorId>(
      objective.checkpoints(), nonGrads));
  summary_.setTargetGrads(
      util::getValues<TensorIds, TensorId>(objective.targets(), grads));
}

Autodiff::Autodiff(const guide::Objective &objective_,
                   const guide::GraphInfo &gi_,
                   GraphMutator &gm_)
    : objective(objective_), graphInfo(gi_), graphMutator(gm_),
      guide(objective, gi_) {
  setNonGrads();
  initGrads();
  setGradsIn();
  addGradsInToGrads();
  backpropagate();
}

std::string Autodiff::genGradInsName(OpId opId,
                                     const std::vector<InIndex> &is) {
  std::ostringstream oss;
  oss << "grad-of-op-" << opId << "-inputs-";
  poprithms::util::append(oss, is);
  return oss.str();
}

std::string Autodiff::genCheckpointName(const TensorId &tId) {
  return "checkpoint/" + tId.str();
}

std::string Autodiff::genRerunName(const OpId opId) {
  return "rerun/" + opId;
}

std::string Autodiff::genInitGradName(const TensorId &tId) {
  return "init-grad-of/" + tId.str();
}

std::string Autodiff::genInGradName(const TensorId &tId) {
  return "grad-in-of/" + tId.str();
}
} // namespace core

} // namespace autodiff
} // namespace poprithms
