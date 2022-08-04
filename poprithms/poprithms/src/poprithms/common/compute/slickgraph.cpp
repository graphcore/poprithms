// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iterator>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/compute/slickgraph.hpp>

namespace poprithms {
namespace common {
namespace compute {

OptionalTensors
SlickConverter::getOptionalTensors(Graph &g,
                                   const OptionalTensorIds &optTenIds) {
  OptionalTensors optTens(optTenIds.size());
  for (uint64_t i = 0; i < optTenIds.size(); ++i) {
    if (optTenIds[i].has_value()) {
      optTens[i] = Tensor(optTenIds[i].value(), &g);
    }
  }
  return optTens;
}

OptionalTensorIds
SlickConverter::getOptionalTensorIds(const OptionalTensors &optTens) {
  // initialize the optional tensor ids to be unset.
  OptionalTensorIds optTenIds(optTens.size());
  for (uint64_t i = 0; i < optTens.size(); ++i) {
    // get the id of the optional tensor ("strip" the graph from it).
    optTenIds[i] = optTens[i];
  }
  return optTenIds;
}

TensorIds SlickConverter::getIds(const Tensors &tensors) {
  TensorIds tensorIds;
  tensorIds.reserve(tensors.size());
  for (const auto &t : tensors) {
    tensorIds.push_back(t.id());
  }
  return tensorIds;
}

SubGraph SlickGraph::createSubGraph(const std::string &sgName) {
  auto sgId = createSubGraphId(sgName);
  return SubGraph(sgId, *this);
}

SubGraphs SlickGraph::createSubGraphs(const std::vector<std::string> &ns) {
  SubGraphs sgs;
  sgs.reserve(ns.size());
  for (auto n : ns) {
    sgs.push_back(createSubGraph(n));
  }
  return sgs;
}

OptionalTensors
SlickGraph::getOptionalTensors(const OptionalTensorIds &optTenIds) {
  return SlickConverter::getOptionalTensors(*this, optTenIds);
}

Tensors SlickGraph::tensors(const TensorIds &tIds) {
  Tensors ts;
  ts.reserve(tIds.size());
  for (auto tId : tIds) {
    ts.push_back(tensor(tId));
  }
  return ts;
}

Tensor SlickGraph::tensor(const TensorId &tId) {
  poprithms::common::multiout::Graph::verifyValidTensorId(tId);
  // A Tensor is constructed from a pair (tensor id, pointer to Graph).
  return Tensor(tId, this);
}

Tensors SlickGraph::tensors(SubGraphId sgId) {
  TensorIds tIds = tensorIds(sgId);
  return tensors(tIds);
}

TensorIds SlickGraph::getIds(const Tensors &tensors) {
  TensorIds tensorIds;
  tensorIds.reserve(tensors.size());
  for (const auto &t : tensors) {
    tensorIds.push_back(t.id());
  }
  return tensorIds;
}

std::ostream &operator<<(std::ostream &ost, const SlickGraph &g) {

  if (!g.isSchedulable()) {
    g.append(ost);
  } else {
    g.appendScheduled(ost);
  }

  return ost;
}

} // namespace compute
} // namespace common
} // namespace poprithms
