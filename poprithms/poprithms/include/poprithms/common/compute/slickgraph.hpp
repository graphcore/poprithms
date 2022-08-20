// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_SLICKGRAPH_HPP
#define POPRITHMS_COMMON_COMPUTE_SLICKGRAPH_HPP

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

class SlickConverter {
public:
  static OptionalTensors
  getOptionalTensors(Graph &g, const OptionalTensorIds &optTenIds);

  /**
   * Get optional tensors ids by extracting them from the optional tensors in
   * #optTensor.
   * */
  static OptionalTensorIds
  getOptionalTensorIds(const OptionalTensors &optTens);

  /**
   * Get the ids of the tensors in #tensors.
   * */
  static TensorIds getIds(const Tensors &tensors);
};

/**
 * Extension to the Graph class which allows constuction with the more 'slick'
 * tensor-centric syntax.
 * */
class SlickGraph : public Graph {
public:
  virtual ~SlickGraph() override = default;
  SlickGraph() : Graph() {}
  SlickGraph(uint64_t n, ReplicationFactor r) : Graph(n, r) {}

  /**
   * Create a sub-graph with name #sgName. The SubGraph class is a thin
   * wrapper around (1) a SubGraphId and a (2) Graph, which acts as
   * syntactic sugar for creating sub-graphs. A SubGraph is to a
   * SubGraphId as a Tensor is to a TensorId.
   * */
  SubGraph createSubGraph(const std::string &sgName);

  SubGraphs createSubGraphs(const std::vector<std::string> &ns);

  SubGraph subGraph(SubGraphId sgId) { return SubGraph(sgId, *this); }

  /**
   * Get optional tensors by combining this graph with the optional tensor ids
   * in #optTenIds.
   * */
  OptionalTensors getOptionalTensors(const OptionalTensorIds &optTenIds);

  /**
   * Get the tensors with ids #tIds.
   *
   * Recall that a tensor is a pair (TensorId, &Graph), so this method just
   * constructs Tensors from the ids #tIds.
   * */
  Tensors tensors(const TensorIds &tIds);

  /**
   * Get a tensor by combining this graph with the tensor id, #tId.
   * */
  Tensor tensor(const TensorId &tId);

  /**
   * Get the tensors in the sub-graph #sgId.
   * */
  Tensors tensors(SubGraphId sgId);

  /**
   * Get the ids of the tensors in #tensors.
   * */
  static TensorIds getIds(const Tensors &tensors);

private:
  virtual void noWeakVTables() override;
};

std::ostream &operator<<(std::ostream &ost, const SlickGraph &g);

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
