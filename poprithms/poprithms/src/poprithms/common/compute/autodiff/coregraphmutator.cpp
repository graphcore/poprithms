// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <map>
#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/common/compute/autodiff/coregraphmutator.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/util/where.hpp>

namespace poprithms {
namespace common {
namespace compute {

TensorId CoreGraphMutator::createZero(const TensorId &tId) {
  return Tensor(tId, &graph_)
      .constant(gradSubGraph, 0.0)
      .expand_(graph_.shape(tId));
}

TensorId CoreGraphMutator::createVariable(const TensorId &like) {
  return Tensor(like, &graph_).variable(gradSubGraph);
}

OpId CoreGraphMutator::clone(OpId opId, const TensorIds &ins) {
  return graph_.clone(opId, ins, gradSubGraph);
}

TensorId CoreGraphMutator::sum(const TensorIds &ts) {
  if (ts.empty()) {
    throw error(
        "poprithms assured that there would be at least 1 tensor in the sum");
  }

  if (ts.size() == 1) {
    return ts[0];
  }

  auto t0 = Tensor(ts.at(0), &graph_);
  for (uint64_t i = 1; i < ts.size(); ++i) {
    t0 = t0 + Tensor(ts.at(i), &graph_);
  }

  return t0.id();
}

OptionalTensorIds
CoreGraphMutator::getInGrads(OpId opId,
                             const autodiff::core::ToGradGraph &toGradGraph) {

  const auto &op_ = graph_.computeOp(opId);

  auto opts = op_.growInGrads(graph_, toGradGraph, gradInfos_, gradSubGraph);

  if (opts.size() != op_.nInTensors()) {
    std::ostringstream oss;
    oss << "The op " << op_ << " has " << op_.nInTensors() << " inputs, but "
        << opts.size() << " (optional) input gradients are created. "
        << op_.nInTensors() << " != " << opts.size() << ".";
    throw error(oss.str());
  }

  return opts;
}

CoreGraphMutator::CoreGraphMutator(
    Graph &m__,
    const autodiff::automatic::GradInfos &gradInfos,
    SubGraphId x__)
    : graph_(m__), gradInfos_(gradInfos), gradSubGraph(x__) {}

} // namespace compute
} // namespace common
} // namespace poprithms
