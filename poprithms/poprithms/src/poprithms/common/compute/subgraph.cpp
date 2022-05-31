// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/rsubgraph_impl.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

template class RSubGraph<Tensor>;

SubGraph::SubGraph(SubGraphId id, Graph &g) : RSubGraph<Tensor>(id, g) {}

} // namespace compute
} // namespace common
} // namespace poprithms
