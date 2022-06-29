// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_GUIDE_GUIDE_HPP
#define POPRITHMS_AUTODIFF_GUIDE_GUIDE_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/autodiff/guide/traversals.hpp>
#include <poprithms/autodiff/ids/ids.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;

/**
 * A class which creates and stores a high-level (calculus-free) description
 * of how differentiation will proceed.
 *
 * Once the Guide is constructed (with the unique constructor below), there
 * are methods which can be queried for high-level information about the flow
 * of gradients in the graph. Examples are nonGradsWithGrads, opsToRerun, etc.
 *
 * */
class Guide {

public:
  /**
   * \param generator The overall objective of the differentiation. contains
   *
   *                  (1) Tensors to target. That is, the non-gradient tensors
   *                      for which gradients are required.
   *
   *                  (2) Tensors to start backpropagation from. This might be
   *                      a 'loss', for example.
   *
   *                  (3) Tensors whose (non-gradient) values are available
   *                      during backpropagation, without needing to be
   *                      recomputed. These are referred to as the
   *                      'checkpoint' tensors.
   *
   * \param graphInfo Describes how each op in the graph is differentiated,
   *                  without any specific calculus details. Specifically,
   *                  which inputs are differentiable with respect to which
   *                  outputs, etc. This defines the overall flow of gradients
   *                  in the DAG.
   *
   * */
  Guide(const Objective &generator, const GraphInfo &graphInfo);

  /**
   * All tensors which have gradients after differentiation. These are:
   * (1) the tensors being targeted
   * (2) the tensors with gradients provided for
   * (3) all input tensors of ops which are differentiated, which are on a
   *     path from a target.
   * (4) all output tensors of ops which are differentiated, through which
   *     gradients propagate. Note that these tensors might not be on a path
   *     to a tensor with a provided gradient.
   * */
  const std::set<TensorId> &nonGradsWithGrads() const {
    return nonGradsWithGrads_;
  }

  bool isNonGradWithGrad(const TensorId &x) const {
    return nonGradsWithGrads_.count(x) != 0;
  }

  /**
   * All ops which must be re-run, as they have at least one output tensor
   * which is not checkpointed and is needed, either directly or indirectly,
   * for differentiating an op. These ops are returned in topologically sorted
   * order.
   **/
  const OpIds &opsToRerun() const { return opsToRerun_; }

  /**
   * A dependency edge map, which specifies constraints on the order in which
   * ops can be differentiated. Keys of the map must be scheduled for
   * differentiation before the corresponding values are. This is the reverse
   * of the order in which the ops appear in the forward (non-gradient) graph.
   * */
  const std::map<OpId, std::set<OpId>> &fwdEdges() const {
    return traversals_.fwdEdges();
  }

  /**
   * The number of ops in #fwdEdges which must be differentiated before
   * another map (the key of this map) can be.
   * */
  std::map<OpId, int64_t> getFwdEdgeDependencyCount() const;

  void append(std::ostream &) const;

  /**
   * The Tensors required to perform differentiation. As an example, suppose
   * there's a 'sin' op which is differentiated, with 'b = sin(a)'. to compute
   * 'dL/da', both 'dL/db' and 'a' are required, and so 'a' will appear in
   * this set. This set will include all non-gradient tensors which are used
   * directly in calculating a gradient. Note that recomputed tensors don't
   * necessarily appear in this set.
   * */
  const std::set<TensorId> &nonGradsForAutodiff() const {
    return nonGradsForAutodiff_;
  }

  const Traversals &traversals() const { return traversals_; }

private:
  /**
   * All the traversals of ops from tensors for which a gradient is required,
   * to tensors with a known gradients, where the backpropagation begins.
   * */
  const OpTraversals &opTraversals() const {
    return traversals_.opTraversals();
  }

  static std::set<OpId> getOps(const OpTraversals &);

private:
  const Objective &generator;
  const GraphInfo &graphInfo;

  // All of these member variables are set in the constructor, and do not
  // change thereafter:
  Traversals traversals_;
  std::set<TensorId> nonGradsForAutodiff_;
  std::set<TensorId> nonGradsWithGrads_;
  std::set<TensorId> nonGradsToRecompute_;
  OpIds opsToRerun_;

  // The steps to constructing the information, all run in the constructor.
  // Information in the implementations.
  void setNonGradsForAutodiff();
  void setNonGradsWithGrads();
  void setNonGradsToRecompute();
  void setOpsToRerun();

  // Run at the end of construction:
  void verifyRecomputeOrder(const GraphInfo &, const Objective &) const;
};

std::ostream &operator<<(std::ostream &, const Guide &);

} // namespace guide
} // namespace autodiff
} // namespace poprithms

#endif
