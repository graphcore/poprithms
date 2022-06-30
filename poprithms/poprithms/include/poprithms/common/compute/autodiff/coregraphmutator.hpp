// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_AUTODIFF_COREGRAPHMUTATOR_HPP
#define POPRITHMS_COMMON_COMPUTE_AUTODIFF_COREGRAPHMUTATOR_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/common/compute/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * An implementation of the core::GraphMutator, specific to a compute::Graph.
 * */
class CoreGraphMutator : public poprithms::autodiff::core::GraphMutator {

private:
  using ToGradGraph = poprithms::autodiff::core::ToGradGraph;
  Graph &graph_;
  const poprithms::autodiff::automatic::GradInfos &gradInfos_;
  SubGraphId gradSubGraph;

public:
  CoreGraphMutator(Graph &,
                   const autodiff::automatic::GradInfos &,
                   SubGraphId gradSubGraph);

  /**
   * Creates a variable like 'like', but create it in #gradSubGraph.
   * */
  TensorId createVariable(const TensorId &like) final;

  /**
   * Create a clone of the op #opId, which has inputs #ins.
   * */
  OpId clone(OpId opId, const TensorIds &ins) final;

  TensorId sum(const TensorIds &) final;

  TensorId createZero(const TensorId &) final;

  void setName(OpId id, const std::string &n) final { graph_.setName(id, n); }

  /**
   * Perform differentiation on the op #opId, to obtain gradients for its
   * inputs. The tensors required to compute these (optional) gradients, such
   * as gradients of the outputs op #opId and inputs and outputs of #opId, are
   * in #toGradGraph.
   * */
  OptionalTensorIds getInGrads(OpId opId,
                               const ToGradGraph &toGradGraph) final;

  std::vector<std::pair<TensorId, TensorId>>
  getCopiesIntoGradOf(const CallEvent &, const ToGradGraph &) const;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
