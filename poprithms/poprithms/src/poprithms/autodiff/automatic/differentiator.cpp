// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/differentiator.hpp>
#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>
#include <poprithms/program/callstack/carriedtensorid.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CarriedTensorId;

void Differentiator::setGrad(OpId callOp, CalleeIndex ci, SubGraphId grad) {
  gradInfos_.setGrad(callOp, ci, grad);
}

Differentiator::~Differentiator() = default;

SubGraphId Differentiator::backwardOutOfGraph(const Objective &o) {
  return backwardOutOfGraph(
      o.gradsProvidedFor(), o.checkpoints(), o.targets());
}

Differentiator::HessianProjections
Differentiator::hvp(const TensorId &source, const TensorIds &targets) {

  auto grads = backward(source, targets);

  auto ddx = cloneWithoutGradInfo();
  TensorIds projectors;
  for (auto x0 : targets) {
    projectors.push_back(mutator_.variableLike(x0, querier_.subGraphId(x0)));
  }

  auto all = targets;
  all.push_back(source);
  auto sg0          = querier_.subGraphIdFromTensorIds(all);
  auto partHessians = ddx->backwardInGraph(
      grads, querier_.tensorIds(sg0), targets, projectors);
  return {partHessians, projectors};
}

Differentiator::CompleteHessian
Differentiator::completeHessian(const TensorId &loss,
                                const TensorId &target) {

  auto hp        = hvp(loss, {target});
  auto sg2       = mutator_.createSubGraphId("sg2");
  auto maskIndex = mutator_.variableLike(loss, DType::Unsigned32, {1});
  auto targetInHessianGraph = mutator_.variableLike(target, sg2);
  auto tShape               = querier_.shape(target);
  auto mask0 = mutator_.variableLike(target, querier_.subGraphId(target));
  mask0      = mutator_.reshape_(mask0, tShape.flatten().unsqueeze(0));
  mask0      = mutator_.encodeOneHot_(mask0, maskIndex);
  mask0      = mutator_.reshape_(mask0, tShape);

  auto constOne =
      mutator_.reshape_(mutator_.scalarConstantLike(maskIndex, 1), {1});
  auto nxtMaskIndex = mutator_.add(maskIndex, constOne);

  auto maskIndex2 = mutator_.variableLike(maskIndex, sg2);
  maskIndex2      = mutator_.zero_(maskIndex2);

  mutator_.removeOp(hp.projections.at(0).opId(),
                    {mask0},
                    "replace projection with delta projection");

  std::vector<CarriedTensorId> cris;
  cris.push_back({targetInHessianGraph, target, target});
  cris.push_back({maskIndex2, maskIndex, nxtMaskIndex});

  auto rpt =
      mutator_.repeat(sg2,
                      querier_.subGraphId(loss),
                      querier_.nelms_u64(target),
                      {},
                      cris,
                      {{hp.projectedTargets.at(0), IsStackedCopy::Yes}},
                      StackedCopyOrder::Up);

  return {sg2, targetInHessianGraph, {rpt, 0}};
}

TensorIds Differentiator::backward(const TensorId &loss,
                                   const TensorIds &vars) {

  for (auto v : vars) {
    if (querier_.subGraphId(v) != querier_.subGraphId(loss)) {
      std::ostringstream oss;
      oss << "The variable being targeted for differentiation, " << v
          << ", is not in the same sub-graph as the loss, " << loss << '.';
      throw error(oss.str());
    }
  }

  if (querier_.nelms_u64(loss) != 1) {
    std::ostringstream oss;
    oss << "Failure in backward(loss=" << loss << ", vars=" << vars
        << "). Expected loss to be a tensor with 1 element, but " << loss
        << " has shape " << querier_.shape(loss)
        << ". Consider sum-reducing it, or use another API "
        << "which allows you to provide an initial gradient. ";
    throw error(oss.str());
  }

  auto outs = backwardInGraph({loss},
                              querier_.tensorIds(querier_.subGraphId(loss)),
                              vars,
                              {mutator_.scalarConstantLike(loss, 1.)});

  return outs;
}

void Differentiator::createMissingGradGraphs(const Objective &objective) {

  poprithms::autodiff::guide::Traversals aTravs(objective, graphInfo());

  for (auto opId : aTravs.traversed()) {
    auto callees_ = querier_.callees(opId);

    if (!callees_.empty()) {

      auto inIndicesTraversed_ = aTravs.inIndicesTraversed(opId);
      for (CalleeIndex ci = 0; ci < callees_.size(); ++ci) {
        if (!gradInfos_.hasGrad(opId, ci)) {

          // If a sub-graph does not have a grad graph, we find or construct
          // one on-the-fly.
          //
          // For the future: it might be better to do this as a separate step
          // (creating and setting gradients) -- in general it is better to
          // know exactly what you're doing at each step, this JIT approach is
          // a bit blind and greedy.

          auto gradsIn = aTravs.outIndicesTraversed(opId);
          InIndices fromTargets;
          for (auto i : inIndicesTraversed_) {
            if (querier_.inDstCalleeIndex(opId, i) == ci) {
              fromTargets.push_back(i);
            }
          }
          const auto loco =
              querier_.localObjective(opId, ci, fromTargets, gradsIn);

          //  A gradient with this objective might have already been created,
          //  for another callee in another op. If not, create one.
          auto cachey = gradInfos().gradGraphsCreatedFor(loco);
          auto sg     = cachey.empty() ? backwardOutOfGraph(loco) : cachey[0];
          setGrad(opId, ci, sg);
        }
      }
    }
  }
}

poprithms::autodiff::core::Summary
Differentiator::getSummary(const Objective &objective, SubGraphId bwd) {

  createMissingGradGraphs(objective);
  auto fwd = querier_.subGraphIdFromObjective(objective);
  auto gm  = graphMutator(bwd);
  poprithms::autodiff::core::Autodiff argoo(objective, graphInfo(), *gm);
  auto summary = argoo.summary();

  insertGradInfo(GradInfo{fwd, bwd, objective, summary});

  for (auto t : objective.targets()) {
    auto gradShape = querier_.shape(gradInfo(bwd).targetGradInGradGraph(t));
    if (gradShape != querier_.shape(t)) {
      std::ostringstream oss;
      oss << "The shape of the gradient is " << gradShape
          << ", which is different to the shape of the target "
          << querier_.shape(t);
      throw error(oss.str());
    }
  }

  auto &&vars = objective.targets();
  auto &&outs = summary.targetGrads();
  if (vars.size() != outs.size()) {
    throw error("Number of targets and target grads are different.");
  }

  for (uint64_t i = 0; i < vars.size(); ++i) {
    if (querier_.shape(vars.at(i)) != querier_.shape(outs.at(i))) {
      std::ostringstream oss;
      oss << "The shape of the gradient is " << querier_.shape(outs.at(i))
          << ", but the shape of the variable is "
          << querier_.shape(vars.at(i));
      throw error(oss.str());
    }
  }

  return summary;
}

void Differentiator::insertGradInfo(const GradInfo &gi) {

  for (auto tId : gi.objective().allTensorIds()) {
    if (querier_.subGraphId(tId) != gi.nonGradSubGraphId()) {
      std::ostringstream oss;
      oss << "The tensor in the autodiff objective " << tId
          << " is in sub-graph " << querier_.subGraphId(tId)
          << ", not the expected non-gradient graph, "
          << gi.nonGradSubGraphId() << '.';
      throw error(oss.str());
    }
  }

  for (auto tId : gi.summary().allTensorIds()) {
    if (querier_.subGraphId(tId) != gi.gradSubGraphId()) {
      std::ostringstream oss;
      oss << "The tensor in the autodiff solution " << tId
          << " is in sub-graph " << querier_.subGraphId(tId)
          << ", not the expected gradient graph, " << gi.gradSubGraphId()
          << '.';
      throw error(oss.str());
    }
  }
  gradInfos_.insert(gi.gradSubGraphId(), gi);
}

TensorIds Differentiator::backwardInGraph(const TensorIds &providedFor,
                                          const TensorIds &cps,
                                          const TensorIds &targs,
                                          const TensorIds &gradsProvided) {

  auto objective = Objective::inGraph(providedFor, cps, targs, gradsProvided);

  const auto sgId =
      querier_.subGraphIdFromTensorIds({providedFor, cps, targs});

  const auto summary = getSummary(objective, sgId);

  return summary.targetGrads();
}

SubGraphId Differentiator::backwardOutOfGraph(const TensorIds &providedFor,
                                              const TensorIds &cps,
                                              const TensorIds &targs) {

  const auto objective = Objective::outOfGraph(providedFor, cps, targs);
  const auto fwd =
      querier_.subGraphIdFromTensorIds({providedFor, cps, targs});
  const auto bwd     = mutator_.createSubGraphId("bwd-of/" + fwd.str());
  const auto summary = getSummary(objective, bwd);

  return bwd;
}

TensorIds Differentiator::minimalNonRecomputationCheckpoints(
    const TensorIds &gradsProvidedFor,
    const TensorIds &targets) {

  // No targets which need gradients? Then no checkpoints are needed.
  if (targets.empty()) {
    return {};
  }

  auto sgId             = querier_.subGraphId(targets[0]);
  const auto objective0 = Objective::outOfGraph(
      gradsProvidedFor, querier_.tensorIds(sgId), targets);
  poprithms::autodiff::guide::Guide guide0(objective0, graphInfo());
  auto cpsSet = guide0.nonGradsForAutodiff();
  TensorIds cps{cpsSet.cbegin(), cpsSet.cend()};
  return cps;
}

void Differentiator::verifyInForwardGraphOf(SubGraphId grad,
                                            const TensorId &inFwd) const {
  const auto &gInfo = gradInfo(grad);
  if (querier_.subGraphId(inFwd) != gInfo.nonGradSubGraphId()) {
    std::ostringstream oss;
    oss << "The non-gradient graph of " << grad << " is "
        << gInfo.nonGradSubGraphId() << ", but the tensor "
        << "in this query, " << inFwd << ", is in "
        << querier_.subGraphId(inFwd) << '.'
        << " It was expected to belong to " << gInfo.nonGradSubGraphId()
        << ". ";
    throw error(oss.str());
  }
}

using GradInPairs     = poprithms::autodiff::core::GradInfo::GradInPairs;
using CheckpointPairs = poprithms::autodiff::core::GradInfo::CheckpointPairs;

} // namespace automatic
} // namespace autodiff
} // namespace poprithms
