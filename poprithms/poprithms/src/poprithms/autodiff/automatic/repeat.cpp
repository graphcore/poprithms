// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/repeat.hpp>
#include <poprithms/common/multiout/skiptraversal.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {
using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CarriedTensorId;

guide::Objective
RepeatDifferentiator::createLocalObjective(const InIndices &fromTargets,
                                           const OutIndices &inGrads) const {

  // The tensors which require gradients. Initialized with external input
  // targets but will be extended for loop carry dependencies.
  std::set<TensorId> targets;
  for (auto i : fromTargets) {
    targets.insert(querier.inDst(rptOpId, i).tId());
  }

  // The tensors which have gradients provided for them. Initialized with
  // external gradients arriving but will be extended for loop carry
  // dependencies.
  std::set<TensorId> gradsProvidedFor;
  for (auto o : inGrads) {
    gradsProvidedFor.insert(querier.outSource(rptOpId, o, CalleeIndex(0)));
  }

  // Perform a traversal to find all tensors on the (loop unrolled) path from
  // #fromTargets to #inGrads.
  const auto visits = gradientPropagationVisits(fromTargets, inGrads);

  // Tensors which are on a (unrolled) path and are sources of carries must be
  // added to the set of tensors which have gradients provided.
  for (const auto &tId : visits) {
    if (repeatQuerier.isCarriedFrom(tId)) {
      gradsProvidedFor.insert(tId);
      targets.insert(repeatQuerier.carriedTo(tId));
    }
  }

  // Checkpoints: All stacked output tensors of the forward op, and all flat
  // outputs which can be verified to have the same value every iteration.
  auto checkpoints = querier.outSources(
      rptOpId, CalleeIndex(0), repeatQuerier.stackedOutIndices());
  for (auto o : repeatQuerier.flatOutIndices()) {
    TensorId outSrc = querier.outSource(rptOpId, o, CalleeIndex(0));
    if (repeatQuerier.definitelySameValueEveryIteration(outSrc)) {
      checkpoints.push_back(outSrc);
    }
  }

  return Objective::outOfGraph(
      TensorIds(gradsProvidedFor.cbegin(), gradsProvidedFor.cend()),
      checkpoints,
      TensorIds(targets.cbegin(), targets.cend()));
}

std::set<TensorId> RepeatDifferentiator::gradientPropagatesFwdFrom(
    const InIndices &inIndices) const {
  auto accept = [this](const auto &x) {
    return querier.gradientPropagates(x);
  };
  using namespace poprithms::common::multiout;
  return depthFirstFwdWithSkips(repeatQuerier,
                                querier,
                                querier.inDsts(rptOpId, inIndices),
                                accept,
                                repeatQuerier.repeatCount());
}

std::set<TensorId> RepeatDifferentiator::gradientPropagatesBwdFrom(
    const OutIndices &outIndices) const {

  using namespace poprithms::common::multiout;
  return depthFirstBwdWithSkips(
      repeatQuerier,
      querier,
      querier.outSources(rptOpId, CalleeIndex(0), outIndices),
      [this](const auto &x) { return querier.gradientPropagates(x); },
      repeatQuerier.repeatCount());
}

TensorIds RepeatDifferentiator::gradientPropagationVisits(
    const InIndices &inIndices,
    const OutIndices &outIndices) const {

  auto bwd = gradientPropagatesBwdFrom(outIndices);
  auto fwd = gradientPropagatesFwdFrom(inIndices);
  std::vector<TensorId> intersection(bwd.cbegin(), bwd.cend());
  for (auto x : fwd) {
    if (bwd.count(x) != 0) {
      intersection.push_back(x);
    }
  }
  return intersection;
}

void IRepeatQuerier::verifyFirstIsSecondStacked(uint64_t rptCnt,
                                                const Shape &stackedShape,
                                                const Shape &unstackedShape) {
  const Shape expectedStackedShape = unstackedShape.prepend(rptCnt);
  if (stackedShape != expectedStackedShape) {
    std::ostringstream oss;
    oss << "Error in verifyFirstIsSecondStacked. 'stacked' shape is "
        << stackedShape << " and 'unstacked' shape is " << unstackedShape
        << ". But with repeat count of " << rptCnt
        << ", the expected stacked shape is " << expectedStackedShape << '.';
    throw error(oss.str());
  }
}

OptionalTensorIds
RepeatDifferentiator::createInGrads(IAutomaticMutator &mutator,
                                    const core::ToGradGraph &toGradGraph,
                                    const GradInfos &gradInfos,
                                    const SubGraphId toExtend) const {

  if (repeatQuerier.stackedCopyOrder() == StackedCopyOrder::Down) {
    std::ostringstream oss;
    oss << "Currently only StackedCopyOrder::Up repeat ops can be "
        << "differentiated. "
        << "It will not be hard to support StackedCopyOrder::Down, "
        << "but it will require associating an order "
        << "to every tensor, not a single one for the op as a while. ";
    throw error(oss.str());
  }

  const auto callGradGraph = gradInfos.grad(rptOpId, CalleeIndex(0));
  const auto caller        = toExtend;
  const auto callee        = callGradGraph;
  const auto &gradInfo_    = gradInfos.at(callGradGraph);

  // The gradient of a repeat op is a new repeat op. We will determine and
  // construct the new repeat op's inputs and outputs:
  std::vector<CarriedTensorId> carriedIns;
  std::vector<std::pair<TensorId, TensorId>> stackedIns;
  std::vector<std::pair<TensorId, IsStackedCopy>> outsOfRepeatGrad;

  const CallEvent fwdCallEvent{
      rptOpId, querier.callee(rptOpId, CalleeIndex(0)), CalleeIndex(0)};

  // 1) checkpoints. These are normally stacked, but can also be flat. If
  // they're flat, the assumption is that they are the same at all iterations
  // of the forward repeat, and so can use just a single (unstacked) value.
  //
  // We do attempt to verify this, but it hard to know with certainty that a
  // value is different between iterations and the user might have better
  // understanding of how ops interact. This error can be disabled if this
  // situation arises.
  for (auto checkpointPair : gradInfo_.checkpointPairs()) {
    const auto dst =
        querier.dstInCaller(checkpointPair.inNonGradGraph, fwdCallEvent);
    const auto from = toGradGraph.getNonGrad(dst);
    const auto to   = checkpointPair.inGradGraph;
    if (querier.shape(from) != querier.shape(to)) {
      IRepeatQuerier::verifyFirstIsSecondStacked(repeatQuerier.repeatCount(),
                                                 querier.shape(from),
                                                 querier.shape(to));
      stackedIns.push_back({from, to});
    } else {
      if (!repeatQuerier.definitelySameValueEveryIteration(
              checkpointPair.inNonGradGraph)) {
        std::ostringstream oss;
        oss << "The tensor " << checkpointPair.inNonGradGraph
            << " is a non-stacked checkpoint, "
            << "but its value might change between iterations. "
            << "Only unchanging tensors can be non-stacked checkpoints. "
            << "This error can be relaxed if you're certain it does not "
               "change, please raise a poprithms issue/ticket. ";
        throw error(oss.str());
      }
      carriedIns.push_back({from, to, to});
    }
  }

  // The input indices which will have a gradient computed for them. Note that
  // this is a superset of the inputs which are traversed through from the
  // original targets, as some inputs which are loop carry destinations will
  // also have gradients computed for them. We expect a pruning pass to remove
  // any ununsed output gradients from the backwards repeat (beyond the scope
  // of this autodd project).
  InIndices fwdTargetedInIndices;
  for (InIndex i = 0; i < querier.nInTensors(rptOpId); ++i) {
    if (gradInfo_.objective().isTarget(querier.inDst(rptOpId, i).tId())) {
      fwdTargetedInIndices.push_back(i);
    }
  }

  // All outputs of the fwd repeat which will have gradients available for the
  // backwards rpt. Note that this is a supet set of all the outputs on a
  // differentiable path to the loss, as it will include some outputs which
  // are carry sources.
  const auto optOuts =
      toGradGraph.optionalGrads(querier.outTensorIds(rptOpId));
  auto hasGradAvailable = [&optOuts](OutIndex o) {
    return optOuts.at(o.get()).has_value();
  };

  // 2) gradient ins. Some of these are stacked, some of them are carried.
  // Some of the carried gradients must be initialized with 0.
  auto getCarrySource =
      [&hasGradAvailable, &toGradGraph, &mutator, this](
          const TensorId &nonGradInNonGradGraph) -> TensorId {
    // The tensor #nonGradInNonGradGraph isn't even an output of #rpt, so it
    // definitely won't have a gradient input to the gradient repeat.
    if (!querier.isOutSource(
            rptOpId, CalleeIndex(0), nonGradInNonGradGraph)) {
      return mutator.zeroLike(
          nonGradInNonGradGraph, querier.subGraphId(rptOpId), "rpt-in-grad");
    }

    // If #nonGradInNonGradGraph is an output of #rpt, but is not on
    // differentiable path to the loss, then there'll be no gradient from
    // loss for it.
    auto index =
        querier.copyOutIndex(rptOpId, CalleeIndex(0), nonGradInNonGradGraph);
    if (!hasGradAvailable(index)) {
      return mutator.zeroLike(
          nonGradInNonGradGraph, querier.subGraphId(rptOpId), "");
    }

    if (repeatQuerier.isStackedOut(nonGradInNonGradGraph)) {
      if (!querier.isDefinitelyAllConstZero(
              toGradGraph.getGrad(querier.dstInCaller(
                  nonGradInNonGradGraph,
                  {rptOpId, querier.callee(rptOpId, 0), 0})))) {
        std::ostringstream oss;
        oss << "\bA Tensor (" << nonGradInNonGradGraph
            << ") in the forward repeat op " << querier.str(rptOpId) << " is"
            << "\n     (1) the source of a carry (copied to another tensor "
            << "for subsequent iteration), and "
            << "\n     (2) a stacked output, and "
            << "\n     (3) it might have a non-zero derivative in the "
            << "backwards pass. "
            << "\nThis becomes a problem with our formulation of "
            << "autodiff. Please ensure one of (1), (2) and (3) is false. "
            << "For example, to make a copy of " << nonGradInNonGradGraph
            << " and make the copy the stacked output. "
            << "Alternatively, manually implement the stacked input with a "
               "dynamic slice. ";
        throw error(oss.str());
      } else {
        return mutator.zeroLike(
            nonGradInNonGradGraph, querier.subGraphId(rptOpId), "");
      }
    }

    // non-zero gradient from loss expected.
    auto x = toGradGraph.getGrad(querier.outTensorId(rptOpId, index));

    return x;
  };

  for (auto x : gradInfo_.gradInPairs()) {
    if (repeatQuerier.isCarriedFrom(x.nonGradInNonGradGraph)) {

      auto tg = gradInfo_.targetGradInGradGraph(
          repeatQuerier.carriedTo(x.nonGradInNonGradGraph));

      carriedIns.push_back(
          {getCarrySource(x.nonGradInNonGradGraph), x.gradInGradGraph, tg});

    }

    else if (repeatQuerier.isStackedOut(x.nonGradInNonGradGraph)) {
      stackedIns.push_back({toGradGraph.getGrad(querier.dstInCaller(
                                x.nonGradInNonGradGraph,
                                {rptOpId, querier.callee(rptOpId, 0), 0})),
                            x.gradInGradGraph});
    }

    // Just a flat output, not carried back:
    else {
      auto finalGrad =
          mutator.unsqueeze_(getCarrySource(x.nonGradInNonGradGraph), 0);
      if (repeatQuerier.repeatCount() > 1) {
        auto zero_ = mutator.zeroLike(finalGrad, querier.subGraphId(rptOpId));
        zero_ = mutator.broadcast_(zero_, repeatQuerier.repeatCount() - 1, 0);
        finalGrad = mutator.concat_({zero_, finalGrad}, 0);
      }

      stackedIns.push_back({finalGrad, x.gradInGradGraph});
    }
  }

  // 4) outputs
  for (auto inIndex : fwdTargetedInIndices) {
    auto gradToCopyOut = gradInfo_.targetGradInGradGraph(
        querier.inDst(rptOpId, inIndex).tId());
    outsOfRepeatGrad.push_back({gradToCopyOut,
                                repeatQuerier.isStackedIn(inIndex)
                                    ? IsStackedCopy::Yes
                                    : IsStackedCopy::No});
  }

  // The gradient repeat iterates through stacked tensors in reverse order
  const auto gradOpDir = StackedCopyOrder::Down;

  const auto bwdRepeat = mutator.repeat(caller,
                                        callee,
                                        repeatQuerier.repeatCount(),
                                        stackedIns,
                                        carriedIns,
                                        outsOfRepeatGrad,
                                        gradOpDir);

  OptionalTensorIds gradsOfIns(querier.nInTensors(rptOpId));
  for (uint64_t o = 0; o < outsOfRepeatGrad.size(); ++o) {
    gradsOfIns[fwdTargetedInIndices[o].get()] = TensorId{bwdRepeat, o};
  }

  return gradsOfIns;
}

} // namespace automatic
} // namespace autodiff
} // namespace poprithms
