// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RSUBGRAPH
#define POPRITHMS_COMMON_COMPUTE_RSUBGRAPH

#include <memory>
#include <vector>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::ndarray::TensorInfo;

class BaseSubGraph {

public:
  SubGraphId id() const { return id_; }

  /**
   * Implictly cast this sub-graph to its sub-graph id.
   * */
  operator SubGraphId() const { return id(); }

  /**
   * Insert a call op into this sub-graph. A call op consists of 3 parts:
   *
   * 1) A set of copies into a callee sub-graph. The pairs of tensors in
   *    #ins each define these copies. Each pair has (1) a source (in this
   *    sub-graph) and (2) a destination (in #callee sub-graph).
   *
   * 2) A sub-graph #callee to run.
   *
   * 3) A set of copies out of #callee. The tensors in #outs (which are
   *    tensors in the #callee sub-graph) are copied into this sub-graph. The
   *    outputs of the returned op are the destination tensors in this
   *    sub-graph of #outs.
   * */
  OpId call(SubGraphId callee,
            const std::vector<std::pair<TensorId, TensorId>> &ins,
            const TensorIds &outs);

  /**
   * Insert a call op into this sub-graph. The callee sub-graph is #callee and
   * the inputs are #ins, as defined in #call. The outputs are all tensors in
   * the sub-graph #callee.
   * */
  OpId callAllOut(SubGraphId callee,
                  const std::vector<std::pair<TensorId, TensorId>> &ins) {
    return call(callee, ins, graph().tensorIds(callee));
  }

protected:
  const Graph &graph() const { return *pGraph_; }
  Graph &graph() { return *pGraph_; }

  BaseSubGraph(SubGraphId id, Graph &graph) : id_(id), pGraph_(&graph) {}

  /**
   * Each op stores the copies into and out of callees it is involved in. This
   * method registers all relevant copies for the op with callees, #opId.
   * */
  void registerCopies(OpId opId);

private:
  BaseSubGraph() = delete;
  SubGraphId id_;
  Graph *pGraph_;
};

/**
 * \tparam T the tensor class of this sub-graph class. Templatizing this class
 *           allows the common::compute project to be extended - a user can
 *           create their own tensor class with an API which integrates custom
 *           ops.
 * */
template <class T> class RSubGraph : public BaseSubGraph {

public:
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

  /**
   * Create a variable tensor (T) in this sub-graph.
   *
   * \param t   The numerical type of the tensor.
   * \param s   Then shape of the tensor.
   * \param d   The device which the tensor is on.
   * */
  T variable(DType t, const Shape &s, DeviceId d);

  T variable(const TensorInfo &info) {
    return variable(info.dtype(), info.shape(), info.deviceId());
  }

  RSubGraph(SubGraphId id, Graph &graph) : BaseSubGraph(id, graph) {}
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
