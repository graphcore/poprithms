// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_SUBGRAPH
#define POPRITHMS_COMMON_COMPUTE_SUBGRAPH

#include <poprithms/common/compute/rsubgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Graph;
class SubGraph;
extern template class RSubGraph<Tensor>;

/**
 * See the RSubGraph template class for information about this class.
 * */
class SubGraph : public RSubGraph<Tensor> {

public:
  SubGraph(SubGraphId, Graph &);
  SubGraph() = delete;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
