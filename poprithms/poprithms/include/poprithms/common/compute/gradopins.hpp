// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_GRADOPINS_HPP
#define POPRITHMS_COMMON_COMPUTE_GRADOPINS_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/common/compute/gradopinids.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

class GradOpIns
    : public poprithms::autodiff::automatic::OpIn<Tensor, OptionalTensor> {

public:
  /**
   * Construct a GradOpIns object from #graph and the tensor ids in #gInIds.
   * The tensors stored in this GradOpIns object are created by combining
   * #graph with each of the tensor ids in #gInIds.
   * */
  GradOpIns(Graph &graph, const GradOpInIds &gInIds)
      : poprithms::autodiff::automatic::OpIn<Tensor, OptionalTensor>(
            SlickConverter::getOptionalTensors(graph, gInIds.getIns()),
            SlickConverter::getOptionalTensors(graph, gInIds.getOuts()),
            SlickConverter::getOptionalTensors(graph,
                                               gInIds.getGradsOfOuts())) {}
};
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
