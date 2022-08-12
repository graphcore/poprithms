// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mock/memory/alias/mockgraph.hpp>

#include <poprithms/memory/alias/graph.hpp>

namespace mock::poprithms::memory::alias {

MockGraph::MockGraph()  = default;
MockGraph::~MockGraph() = default;

MockGraph *mockAliasGraph_ = nullptr;

} // namespace mock::poprithms::memory::alias

namespace poprithms::memory::alias {
TensorId Graph::allocate(const Shape &shape, Color color) {
  return mock::poprithms::memory::alias::mockAliasGraph_->allocate(shape,
                                                                   color);
}

TensorId Graph::clone(TensorId toCloneId, CloneColorMethod cloneColorMethod) {
  return mock::poprithms::memory::alias::mockAliasGraph_->clone(
      toCloneId, cloneColorMethod);
}
} // namespace poprithms::memory::alias
