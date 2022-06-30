// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_SLICKGRAPH_HPP
#define POPRITHMS_COMMON_COMPUTE_SLICKGRAPH_HPP

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/common/compute/tslick.hpp>

namespace poprithms {
namespace common {
namespace compute {

using SlickGraph =
    poprithms::common::compute::TSlickGraph<Tensor, OptionalTensor, SubGraph>;

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
