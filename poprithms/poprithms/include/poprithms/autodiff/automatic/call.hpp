// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_CALL_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_CALL_HPP

#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

class CallDifferentiator {

public:
  /**
   * Create input gradients for the call op #callOpId by extending the
   * sub-graph #toExtend with a call to a gradient sub-graph.
   * */
  static OptionalTensorIds
  createInGrads(OpId callOpId,
                IAutomaticMutator &,
                const IAutomaticQuerier &,
                const poprithms::autodiff::core::ToGradGraph &,
                const GradInfos &,
                SubGraphId toExtend);
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
