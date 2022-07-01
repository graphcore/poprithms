// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_CORE_AUTODIFF_HPP
#define POPRITHMS_AUTODIFF_CORE_AUTODIFF_HPP

#include <map>
#include <string>
#include <vector>

#include <poprithms/autodiff/core/graphmutator.hpp>
#include <poprithms/autodiff/core/summary.hpp>
#include <poprithms/autodiff/core/togradgraph.hpp>
#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

/**
 * The main class for differentiating a graph.
 * */
class Autodiff : public ToGradGraph {

public:
  /**
   * The Autodiff constructor, where the gradient graph is created.
   *
   * \param objective  Describes how to differentiate the graph (what are the
   *                   targets? what are the checkpoints? what are the sources
   *                   of the gradients to propagate?).
   *
   * \param graphInfo  Describes the DAG structure of the graph to
   *                   differentiate, and how gradients flow through ops.
   *
   * \param mutator  Describes the 'calculus' of the ops, and how to do
   *                 certain generic tasks such as add 2 tensors, and create a
   *                 zero tensor.
   * */
  Autodiff(const guide::Objective &objective,
           const guide::GraphInfo &graphInfo,
           GraphMutator &mutator);

  /**
   * Get a summary of the differentiation. \sa Summary.
   * */
  const Summary &summary() const { return summary_; }

private:
  const guide::Objective &objective;
  const guide::GraphInfo &graphInfo;
  GraphMutator &graphMutator;
  guide::Guide guide;

  OptionalTensorId optionalGrad(const TensorId &) const final;
  OptionalTensorId optionalNonGrad(const TensorId &) const final;

  OptionalTensorIds optionalNonGrads(const TensorIds &ids) const final;
  OptionalTensorIds optionalGrads(const TensorIds &ids) const final;

  // The steps for doing autodifferentiation.
  void setNonGrads();
  void initPartialGradsToBeSummed();
  void setGradsIn();
  void addGradsInToGrads();
  void backpropagate();

  Summary summary_;

  std::map<TensorId, TensorId> nonGrads;
  std::map<TensorId, TensorIds> partialGradsToBeSummed;

  // The final summations of the vectors in partialGradsToBeSummed:
  std::map<TensorId, TensorId> grads;
  std::map<TensorId, TensorId> gradsIn;

  // insert 'grad' into the vector of partial tensors of 'nonGrad' in
  // partialGradsToBeSummed.
  void registerPartialGrad(const TensorId &nonGrad, const TensorId &grad);

  // insert a gradient into 'grads', which is the sum of partial gradients in
  // 'partialGradsToBeSummed'.
  void setGradFromPartials(const TensorId &nonGrad);

public:
  // Methods for creating debug name strings. These are not used in any logic,
  // only for logging and testing purposed.
  static std::string genGradInsName(OpId opId,
                                    const std::vector<InIndex> &is);

  static std::string genCheckpointName(const TensorId &tId);

  static std::string genRerunName(const OpId opId);

  static std::string genInitGradName(const TensorId &tId);

  static std::string genInGradName(const TensorId &tId);
};

} // namespace core
} // namespace autodiff
} // namespace poprithms

#endif
