// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <autodiff/autodiff/error.hpp>
#include <map>
#include <sstream>

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
 * Initialize the partial gradient Tensor vectors.
 *
 * For all the non-gradient tensors which require a gradient at some point
 * during differentiation, a vector is kept to store the partial gradients.
 * This method initializes all these vectors as empty vectors. At some point
 * when the vector is fully populated, that is when all the partial gradients
 * required to compute a final gradient have been obtained, a final gradient
 * is computed (using the user implemented 'sum' method if the vector is
 * non-empty, or the user's implemented 'zero' method od the vector is empty).
 * This is done in the method 'setGradFromPartials'
 * */
void Autodiff::initPartialGradsToBeSummed() {
  for (const auto &tId : guide.nonGradsWithGrads()) {
    partialGradsToBeSummed.insert({tId, {}});
  }
}

TensorId Autodiff::getGrad(const TensorId &nonGrad) const {
  const auto found = grads.find(nonGrad);
  if (found == grads.cend()) {
    throw error("No grad for " + nonGrad.str() + " found, in getGradIdRef. ");
  }
  return found->second;
}

void Autodiff::registerPartialGrad(const TensorId &nonGrad,
                                   const TensorId &grad) {
  const auto found = partialGradsToBeSummed.find(nonGrad);
  if (found == partialGradsToBeSummed.cend()) {
    throw error(std::string("No vector of partial gradients initialized ") +
                "for the non-gradient tensor " + nonGrad.str() +
                ". Are you sure this non-gradient " +
                "tensor requires a gradient at any point?");
  }
  found->second.push_back(grad);
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

  const auto getGrad_ = [this](uint64_t i) {
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
    gradsIn.insert({id, getGrad_(i)});
  }
}

// Add input gradients to gradients.
void Autodiff::addGradsInToGrads() {
  for (const auto &[gradProvidedFor, gradIn] : gradsIn) {
    registerPartialGrad(gradProvidedFor, gradIn);
  }
}

void Autodiff::setGradFromPartials(const TensorId &outId) {
  auto found = partialGradsToBeSummed.find(outId);
  if (found == partialGradsToBeSummed.cend()) {
    std::ostringstream oss;
    oss << "Failed to find an initialised (even empty!) "
        << "vector of partial gradients for the non-grad tensor, "
        << outId.str()
        << ". Can therefore not create a final gradient for this tensor "
        << "(are you sure it needs a gradient at some point?)";
    throw error(oss.str());
  }

  // The sum of zero tensors is the zero tensor:
  if (found->second.size() == 0) {
    auto tId = graphMutator.createZero(outId);
    graphMutator.setName(tId.opId(), genInitGradName(outId));
    grads.insert({outId, tId});
  } else {
    const auto sumTensorId = graphMutator.sum(found->second);
    grads.insert({outId, sumTensorId});
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

    for (auto outId : graphInfo.outTensorIds(nxt)) {
      if (guide.isNonGradWithGrad(outId)) {
        setGradFromPartials(outId);
      }
    }

    // To populate: the gradients of each of the inputs of 'nxt':
    const OptionalTensorIds inGrads = graphMutator.getInGrads(nxt, *this);

    std::map<OpId, std::vector<InIndex>> creates;

    for (uint64_t i = 0; i < graphInfo.nInTensors(nxt); ++i) {
      if (inGrads.at(i).has_value()) {
        creates[inGrads.at(i).value().opId()].push_back(i);
        const auto inId = graphInfo.inTensorId(nxt, InIndex(i));
        if (partialGradsToBeSummed.count(inId) != 0) {
          registerPartialGrad(inId, inGrads[i].value());
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

  for (const auto &[tId, ts] : partialGradsToBeSummed) {
    (void)ts;
    if (grads.find(tId) == grads.cend()) {
      setGradFromPartials(tId);
    }
  }

  summary_ = Summary(
      util::getValues<TensorIds, TensorId>(objective.gradsProvidedFor(),
                                           gradsIn),
      util::getValues<TensorIds, TensorId>(objective.checkpoints(), nonGrads),
      util::getValues<TensorIds, TensorId>(objective.targets(), grads));
}

Autodiff::Autodiff(const guide::Objective &objective_,
                   const guide::GraphInfo &gi_,
                   GraphMutator &gm_)
    : objective(objective_), graphInfo(gi_), graphMutator(gm_),
      guide(objective, gi_) {
  setNonGrads();
  initPartialGradsToBeSummed();
  setGradsIn();
  addGradsInToGrads();
  backpropagate();
}

std::string Autodiff::genGradInsName(OpId opId,
                                     const std::vector<InIndex> &is) {
  auto sortedIs = is;
  std::sort(sortedIs.begin(), sortedIs.end());
  std::ostringstream oss;
  oss << "grad-of-op-" << opId << "-inputs-";
  poprithms::util::append(oss, sortedIs);
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
