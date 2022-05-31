// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RSUBGRAPH
#define POPRITHMS_COMMON_COMPUTE_RSUBGRAPH

#include <memory>
#include <vector>

#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;

class Graph;

/**
 * \tparam T the tensor class of this sub-graph class. Templatizing this class
 *           allows the common::compute project to be extended - a user can
 *           create their own tensor class with an API which integrates custom
 *           ops.
 * */
template <class T> class RSubGraph {

public:
  SubGraphId id() const { return id_; }

  /**
   * Implictly cast this sub-graph to its sub-graph id.
   * */
  operator SubGraphId() const { return id(); }

  /**
   * Create a constant tensor (T) in this sub-graph
   *
   * \param v   The value of the tensor.
   * \param d   The device which the constant is on.
   *
   * \return A constant tensor (T).
   * */
  T constant(const HostTensor &v, DeviceId d);

  /**
   * Create a scalar constant tensor, of type #t and value #v, on device #d.
   * */
  T constant(DType t, double v, DeviceId d) {
    return constant(HostTensor::scalar(t, v), d);
  }

  RSubGraph(SubGraphId id, Graph &graph) : id_(id), pGraph_(&graph) {}

protected:
  const Graph &graph() const { return *pGraph_; }
  Graph &graph() { return *pGraph_; }

private:
  RSubGraph() = delete;
  SubGraphId id_;
  Graph *pGraph_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
