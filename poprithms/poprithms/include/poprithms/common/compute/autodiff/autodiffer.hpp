// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTODIFFER_HPP
#define POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTODIFFER_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

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
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
