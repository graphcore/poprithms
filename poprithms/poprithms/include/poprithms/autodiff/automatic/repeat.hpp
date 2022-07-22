// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_REPEAT_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_REPEAT_HPP

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

/**
 * Interface class for a repeat op.
 * */
class IRepeatQuerier {

public:
  virtual ~IRepeatQuerier() = default;

  /**
   * Assuming that #tId is an input with a loop carry dependency, what tensors
   * in the callee sub-graph is it copied from?
   * */
  virtual TensorId carriedTo(const TensorId &) const = 0;

  /**
   * Return true if #tId is an input with a loop carry dependency.
   * */
  virtual bool isCarriedTo(const TensorId &tId) const = 0;

  /**
   * The inverse of #isCarriedTo and #carriedTo.
   * */
  virtual bool isCarriedFrom(const TensorId &) const   = 0;
  virtual TensorId carriedFrom(const TensorId &) const = 0;

  /**
   * We currently assume that all stacked inputs and outputs are iterated
   * through in the same order (ascending or descending) although there is a
   * task to allow for different directions. TODO(T66493).
   * */
  virtual StackedCopyOrder stackedCopyOrder() const = 0;

  /**
   * Return true if the #i'th input is stacked.
   * */
  virtual bool isStackedIn(InIndex i) const = 0;

  /**
   * Return true if #tId is a stacked output in the callee sub-graph.
   * */
  virtual bool isStackedOut(const TensorId &) const = 0;

  /**
   * The number of iterations the callee sub-graph is executed.
   * */
  virtual uint64_t repeatCount() const = 0;

  /**
   * Return all indices for which outputs are stacked (all values at all
   * iterations are output).
   * */
  virtual OutIndices stackedOutIndices() const = 0;

  /**
   * Return all indices for which outputs are not stacked (only the value from
   * the final iteration is returned).
   * */
  virtual OutIndices flatOutIndices() const = 0;

  /**
   * Throw an error if any of
   *  (1) #s0 has rank 1 higher than #s1,
   *  (2) The first dimension of #s0 is rptCount,
   *  (3) The subseqent dimensions of #s0 are equal to #s1,
   * are not satisfied.
   * */
  static void
  verifyFirstIsSecondStacked(uint64_t, const Shape &s0, const Shape &s1);

  /**
   * \return true if it can be confirmed that the value of #tId does not
   *         change between iterations.
   * */
  virtual bool definitelySameValueEveryIteration(const TensorId &) const = 0;
};

/**
 * Utility class for differentiating a repeat op.
 * */
class RepeatDifferentiator {

private:
  // The id fp the repeat op (in the graph being differentiated):
  OpId rptOpId;

  // Objects for getting information about ops in the graph:
  const IRepeatQuerier &repeatQuerier;
  const IAutomaticQuerier &querier;

public:
  RepeatDifferentiator(OpId a,
                       const IRepeatQuerier &b,
                       const IAutomaticQuerier &c)
      : rptOpId(a), repeatQuerier(b), querier(c) {}

  /**
   * Perform automatic differentation on the op #rptOpId.
   *
   * The created gradient op, which is itself a repeat op, will be inserted
   * into the graph #toExtend.
   * */
  OptionalTensorIds createInGrads(IAutomaticMutator &,
                                  const core::ToGradGraph &,
                                  const GradInfos &,
                                  SubGraphId toExtend) const;

  /**
   * For call ops this method is trivial, as there are 1:1 correspondences
   * between the indices #fromTargets and the targets of differentiation and
   * between #inGrads and the set of outputs which have gradients provided.
   *
   * But for the repeat ops, the sets of tensors might need extension due to
   * loop carry dependencies. This method implements this logic.
   * */
  guide::Objective createLocalObjective(const InIndices &fromTargets,
                                        const OutIndices &inGrads) const;

  /**
   * Traverse through the unrolled callee graph, starting from #inIndices,
   * traversing to all differentiable outputs.
   * */
  std::set<TensorId>
  gradientPropagatesFwdFrom(const InIndices &inIndices) const;

  /**
   * Traverse backwards through the unrolled callee graph starting from
   * #outIndices, traversing to all differentiable inputs (defined by
   * #querier).
   * */
  std::set<TensorId>
  gradientPropagatesBwdFrom(const OutIndices &outIndices) const;

  /**
   * The intersection of all tensors visited both by forward traversal from
   * #inIndices and backwards traversal from #outIndices.
   * */
  TensorIds gradientPropagationVisits(const InIndices &inIndices,
                                      const OutIndices &outIndices) const;
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
