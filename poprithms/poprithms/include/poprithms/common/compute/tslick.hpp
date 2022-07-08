// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TSLICK_HPP
#define POPRITHMS_COMMON_COMPUTE_TSLICK_HPP

#include <poprithms/common/compute/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

class TSlickConverter {
public:
  template <typename TOptionalTensors, typename TTensor>
  static TOptionalTensors
  getOptionalTensors(Graph &g, const OptionalTensorIds &optTenIds) {
    TOptionalTensors optTens(optTenIds.size());
    for (uint64_t i = 0; i < optTenIds.size(); ++i) {
      if (optTenIds[i].has_value()) {
        optTens[i] = TTensor(optTenIds[i].value(), &g);
      }
    }
    return optTens;
  }

  /**
   * Get optional tensors ids by extracting them from the optional tensors in
   * #optTensor.
   * */
  template <typename TOptionalTensors>
  static OptionalTensorIds
  getOptionalTensorIds(const TOptionalTensors &optTens) {
    // initialize the optional tensor ids to be unset.
    OptionalTensorIds optTenIds(optTens.size());
    for (uint64_t i = 0; i < optTens.size(); ++i) {
      // get the id of the optional tensor ("strip" the graph from it).
      optTenIds[i] = optTens[i];
    }
    return optTenIds;
  }

  /**
   * Get the ids of the tensors in #tensors.
   * */
  template <typename TTensors>
  static TensorIds getIds(const TTensors &tensors) {
    TensorIds tensorIds;
    tensorIds.reserve(tensors.size());
    for (const auto &t : tensors) {
      tensorIds.push_back(t.id());
    }
    return tensorIds;
  }
};

/**
 * Extension to the Graph class which allows constuction with the more 'slick'
 * tensor-centric syntax.
 * */
template <class TTensor, class TOptionalTensor, class TSubGraph>
class TSlickGraph : public Graph {
public:
  using TTensors         = std::vector<TTensor>;
  using TSubGraphs       = std::vector<TSubGraph>;
  using TOptionalTensors = std::vector<TOptionalTensor>;

  virtual ~TSlickGraph() override = default;
  TSlickGraph() : Graph() {}
  TSlickGraph(uint64_t n, ReplicationFactor r) : Graph(n, r) {}

  /**
   * Create a sub-graph with name #sgName. The SubGraph class is a thin
   * wrapper around (1) a SubGraphId and a (2) Graph, which acts as
   * syntactic sugar for creating sub-graphs. A SubGraph is to a
   * SubGraphId as a Tensor is to a TensorId.
   * */
  TSubGraph createSubGraph(const std::string &sgName) {
    auto sgId = createSubGraphId(sgName);
    return TSubGraph(sgId, *this);
  }

  TSubGraphs createSubGraphs(const std::vector<std::string> &ns) {
    TSubGraphs sgs;
    sgs.reserve(ns.size());
    for (auto n : ns) {
      sgs.push_back(createSubGraph(n));
    }
    return sgs;
  }

  TSubGraph subGraph(SubGraphId sgId) { return TSubGraph(sgId, *this); }

  /**
   * Get optional tensors by combining this graph with the optional tensor ids
   * in #optTenIds.
   * */
  TOptionalTensors getOptionalTensors(const OptionalTensorIds &optTenIds) {
    return TSlickConverter::getOptionalTensors(*this, optTenIds);
  }

  /**
   * Get tensors by combining this graph with the tensor ids in #tIds.
   * */
  TTensors tensors(const TensorIds &tIds) {
    TTensors ts;
    ts.reserve(tIds.size());
    for (auto tId : tIds) {
      ts.push_back(tensor(tId));
    }
    return ts;
  }

  /**
   * Get a tensor by combining this graph with the tensor id, #tId.
   * */
  TTensor tensor(const TensorId &tId) {
    poprithms::common::multiout::Graph::verifyValidTensorId(tId);
    // A TTensor is constructed from a pair (tensor id, pointer to Graph).
    return TTensor(tId, this);
  }

  /**
   * Get the tensors in the sub-graph #sgId.
   * */
  TTensors tensors(SubGraphId sgId) {
    TensorIds tIds = tensorIds(sgId);
    return tensors(tIds);
  }

  /**
   * Get the ids of the tensors in #tensors.
   * */
  static TensorIds getIds(const TTensors &tensors) {
    TensorIds tensorIds;
    tensorIds.reserve(tensors.size());
    for (const auto &t : tensors) {
      tensorIds.push_back(t.id());
    }
    return tensorIds;
  }
};

template <class TTensor, class TOptionalTensor, class TSubGraph>
inline std::ostream &
operator<<(std::ostream &ost,
           const TSlickGraph<TTensor, TOptionalTensor, TSubGraph> &g) {

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

#endif
