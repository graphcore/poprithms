// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_GUIDE_TRAVERSALS_HPP
#define POPRITHMS_AUTODIFF_GUIDE_TRAVERSALS_HPP

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
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
 * Summary of the ops and tensors traversed to obtain the target gradients of
 * an objective.
 * */
class Traversals {
public:
  Traversals(const Objective &objective, const GraphInfo &);

  /**
   * Traversal order dependencies (from forward graph outputs to inputs).
   * */
  const std::map<OpId, std::set<OpId>> &fwdEdges() const { return fwdEdges_; }

  bool isTraversed(OpId opId) const {
    return fwdEdges_.find(opId) != fwdEdges_.cend();
  }

  const OpIds &traversed() const { return traversed_; }

  void append(std::ostream &) const;

  const OpTraversals &opTraversals() const { return opTraversals_; }

  /**
   * The input indices of #opId which lie on differentiable paths traversed.
   * */
  InIndices inIndicesTraversed(OpId) const;

  /**
   * The output indices of #opId which lie on differentiable paths
   * traversed.
   * */
  OutIndices outIndicesTraversed(OpId) const;

private:
  void setTraversals(const Objective &, const GraphInfo &);
  void setFwdEdges(const GraphInfo &);

  OpTraversals opTraversals_;
  std::unordered_map<OpId, InIndices> inIndicesTraversed_;
  std::unordered_map<OpId, OutIndices> outIndicesTraversed_;
  OpIds traversed_;
  std::map<OpId, std::set<OpId>> fwdEdges_;
};

std::ostream &operator<<(std::ostream &, const Traversals &);

} // namespace guide
} // namespace autodiff
} // namespace poprithms

#endif
