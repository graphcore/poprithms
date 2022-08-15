// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTODIFFER_HPP
#define POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTODIFFER_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <poprithms/autodiff/automatic/autogradfunction.hpp>
#include <poprithms/autodiff/automatic/differentiator.hpp>
#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/common/compute/autodiff/automaticmutator.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Completion of the automatic::Differentiator interface.
 * */
template <class TGraph            = SlickGraph,
          class TAutomaticMutator = AutomaticMutator>
class Autodiffer : public poprithms::autodiff::automatic::Differentiator {

private:
  TGraph &m;
  GuideGraphInfo graphInfo_;
  AutomaticQuerier querier_;
  TAutomaticMutator mutator_;

  const poprithms::autodiff::guide::GraphInfo &graphInfo() const final {
    return graphInfo_;
  }

  std::unique_ptr<poprithms::autodiff::core::GraphMutator>
  graphMutator(SubGraphId sgId) const final {
    return std::unique_ptr<poprithms::autodiff::core::GraphMutator>(
        new CoreGraphMutator(m, gradInfos(), sgId));
  }

  std::unique_ptr<poprithms::autodiff::automatic::Differentiator>
  cloneWithoutGradInfo() const final {
    return std::unique_ptr<Autodiffer>(new Autodiffer(m));
  }

public:
  Autodiffer(TGraph &m_)
      : poprithms::autodiff::automatic::Differentiator(querier_, mutator_),
        m(m_), graphInfo_(m_, gradInfos()), querier_(m_), mutator_(m_) {}

  TGraph &graph() { return m; }
};

/**
 * A thin wrapper above the AutogradFunction method in the autodiff project,
 * which provides a Tensor-centric interface ontop of the more succinct
 * TensorId-centric interface.
 *
 * See autodiff::automatic::AutogradFunction for more detailed descriptions of
 * how this class works.
 * */
class AutogradFunction
    : public poprithms::autodiff::automatic::AutogradFunction {

public:
  AutogradFunction(Autodiffer<SlickGraph> &ad)
      : poprithms::autodiff::automatic::AutogradFunction(ad), ad_(ad) {}

  Tensors operator()(const Tensors &ins, const std::string &dbgString = "") {
    auto ids = apply(Tensor::tensorIds(ins), dbgString);
    return Tensor::tensors(ids, g());
  }

private:
  virtual Tensors fwd(const Tensors &ts) = 0;

  TensorIds forwards(const TensorIds &tIds) final {
    return Tensor::tensorIds(fwd(g().tensors(tIds)));
  }

  virtual OptionalTensors bwd(const Tensors &fwdOuts,
                              const OptionalTensors &fwdOutGrads) = 0;

  OptionalTensorIds backwards(const TensorIds &fwdOuts,
                              const OptionalTensorIds &fwdOutGrads) final {
    return OptionalTensor::fromOptionalTensors(
        bwd(g().tensors(fwdOuts), g().getOptionalTensors(fwdOutGrads)));
  }

  Autodiffer<SlickGraph> &ad_;
  SlickGraph &g() { return ad_.graph(); }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
