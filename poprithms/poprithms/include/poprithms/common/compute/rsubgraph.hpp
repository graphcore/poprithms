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
#include <poprithms/program/callstack/carriedtensorid.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/program/callstack/stackedio.hpp>
#include <poprithms/program/callstack/stackutil.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;
using poprithms::common::compute::RemoteOptions;
using poprithms::common::multiout::ConsumptionId;
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpTraversal;
using poprithms::common::multiout::OpTraversals;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::ndarray::TensorInfo;
using poprithms::program::callstack::CalleeIndex;
using poprithms::program::callstack::CalleeTensorId;
using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CarriedTensorId;
using poprithms::program::callstack::CarriedTensorIds;
using poprithms::program::callstack::CopyIn;
using poprithms::program::callstack::CopyIns;
using poprithms::program::callstack::CopyOuts;
using poprithms::program::callstack::IsStackedCopy;
using poprithms::program::callstack::StackedCopyOrder;

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

  /**
   * \return All ConstInit ops in this sub-graph.
   * */
  OpIds constInitIds() const;

  /**
   * \return All VarInit ops in this sub-graph.
   * */
  OpIds varInitIds() const;

  /**
   * \return All Init (ConstInit and VarInint) ops in this sub-graph.
   * */
  OpIds initIds() const;

  /**
   * \return The ids of all tensors in this sub-graph.
   * */
  TensorIds tensorIds() const;

  TensorIds tensorIds(DeviceType) const;

  /**
   * Insert a repeat op into this sub-graph.
   *
   * \param callee The callee sub-graph to run multiple times.
   *
   * \param repeatCount The number of iterations to run the callee sub-graph.
   *
   * \param stackedInputs Inputs for which there is 1 value (slice) per
   *        iteration. If the tensor in the callee sub-graph has shape #s,
   *        then the shape of the tensor in this sub-graph is (repeatCount,
   *        *s). At each iteration, a slice from the tensor in the (the
   *        caller) sub-graph is copied to the callee tensor. The order in
   *        which the slices are iterated through is controlled by
   *        #stackedCopyOrder. Each element in stackedInputs is a pair, with
   *        element #0 being a stacked tensor in this sub-graph and element #1
   *        being the target of the input copy in sub-graph #callee.
   *
   * \param carriedTensors These are the non-stacked inputs to the callee. See
   *                       the CarriedTensorId class for more information.
   *
   * \param outputs The tensors in the callee sub-graph to be copied out after
   *                the final iteration of the callee sub-graph. The outputs
   *                can either be stacked, which means all of the values from
   *                every iteration is copied out, or not, which means only
   *                the final value of the callee tensor after the final
   *                iteration is copied out.
   *
   * \param stackedCopyOrder All stacked input and output tensors are iterated
   *        through in the same direction: either from index 0 to index
   *        repeatCount - 1 if stackedCopyOrder is StackedCopyOrder::Up, or
   *        from index repeatCount -1 to index 0 if stackedCopyOrder is
   *        StackedCopyOrder::Down.
   * */
  OpId repeat(SubGraphId callee,
              uint64_t repeatCount,
              const std::vector<std::pair<TensorId, TensorId>> &stackedInputs,
              const CarriedTensorIds &carriedTensors,
              const std::vector<std::pair<TensorId, IsStackedCopy>> &outputs,
              StackedCopyOrder stackedCopyOrder = StackedCopyOrder::Up);

  /**
   * Insert a repeat op into this sub-graph. All tensors in the callee
   * sub-graph are copied out, if possible. Having all callee tensors copied
   * out makes it easy to backpropagate through a repeat op without manually
   * listing all checkpoint tensors required.
   *
   * Specifically, let #nonFlatOuts be the set of all tensors in the callee,
   * except for user provided #flatOutputs. That is,
   *
   * nonFlatOuts = {all tensors in callee} \ flatOutputs.
   *
   * Then all tensors in #nonFlatOuts are copied out of the callee, as either
   * (1) stacked tensors if they are not carry sources in #carriedTensors and
   * as (2) flat tensors if they are.
   *
   * For information on arguments #callee, #repeatCount, #stackedInputs,
   * #carriedTensors, and #stackedCopyOrder, see #repeat.
   *
   *
   * \param flatOutputs These are the outputs which are not stacked. That is,
   *                    only the value from the final iteration is returned.
   *
   * */
  OpId repeatAllOut(
      SubGraphId callee,
      uint64_t repeatCount,
      const std::vector<std::pair<TensorId, TensorId>> &stackedInputs,
      const CarriedTensorIds &carriedTensors,
      const TensorIds &flatOutputs,
      StackedCopyOrder stackedCopyOrder = StackedCopyOrder::Up);

  /**
   * \param callees The sub-graphs to switch between, depending on on the
   *                runtime value of #condition.
   *
   * \param condition The 1-element integral tensor which determines which of
   *                  the callees to execute.
   *
   * \param ins The inputs to to the callees. For every input index, there is
   *            (1) a source tensor in the calling graph and
   *            (2) a destination tensor in 1 of the sub-graphs at
   *            (3) a callee index.
   *            The copy from (1) to (2) is only performed when the value of
   *            #condition is (3).
   *
   * \param completeOuts indexed by [outputIndex][calleeIndex] these are
   *        outputs which every callee provided a tensor for. This is in
   *        contrast to #partialOuts, which are outputs which not every callee
   *        needs to provide an output for.
   *
   * \param partialOuts a vector of the partial outputs. Example: Suppose for
   *        example that callees is of size 6 and partialOuts[i] =
   *        {(TensorId=7, CalleeIndex=2), (TensorId=5, CalleeIndex=4)}. Then,
   *        only when callee #2 or callee #4 are called will an output be
   *        copied (otherwise the output copy does not happen).
   **/

  OpId switchOp(
      const SubGraphIds &callees,
      const TensorId &condition,
      const std::vector<std::tuple<TensorId, TensorId, CalleeIndex>> &ins,
      const std::vector<TensorIds> &completeOuts,
      const std::vector<CalleeTensorIds> &unmergedOuts = {});

  /**
   * A switch operation (\sa #switchOp) where all tensors not in #completeOuts
   * is in #unmergedOuts.
   * */
  OpId switchAllOut(
      const SubGraphIds &callees,
      const TensorId &condition,
      const std::vector<std::tuple<TensorId, TensorId, CalleeIndex>> &ins,
      const std::vector<TensorIds> &completeOuts);

  /**
   * Append a summary of this sub-graph to #ost.
   **/
  void append(std::ostream &ost) const;

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

  using Ts = std::vector<T>;

public:
  RSubGraph(SubGraphId id, Graph &graph) : BaseSubGraph(id, graph) {}
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

  /**
   * Create a variable tensor in this sub-graph with tensor information #info.
   * */
  T variable(const TensorInfo &info) {
    return variable(info.dtype(), info.shape(), info.deviceId());
  }

  /**
   * Create a variable tensor in this sub-graph on host.
   * */
  T hostVariable(DType t, const Shape &s) {
    return variable(t, s, graph().host());
  }

  /**
   * Create a variable tensor in this sub-graph which has the float32
   * numerical type, and is the root ipu. The shape of the new variable is #s.
   * */
  T rootIpuFloat32Variable(const Shape &s) {
    return variable(DType::Float32, s, graph().rootIpu());
  }

  /**
   * Create a host double (float64) tensor.
   * */
  T hostFloat64Variable(const Shape &s) {
    return hostVariable(DType::Float64, s);
  }

  /**
   * Create a host float (float32) tensor.
   * */
  T hostFloat32Variable(const Shape &s) {
    return hostVariable(DType::Float32, s);
  }

  /**
   * Create a host int32 variable.
   * */
  T hostInt32Variable(const Shape &s) {
    return hostVariable(DType::Int32, s);
  }

  /**
   * Create a float32 constant.
   * */
  T float32Constant(float v, DeviceId dId) {
    return constant(poprithms::compute::host::Tensor::float32(v), dId);
  }

  /**
   * Create multiple variable tensors in this sub-graph, of numerical type
   * #dtype and on device #devId. The shapes of the created are #shapes.
   * */
  Ts variables(DType dtype, const Shapes &shapes, DeviceId deviId);

  /**
   * Create variables like those in #likeThese. The i'th returned tensor has
   * shape, dtype, and device id that are the same as the i'th tensor in
   * #likeThese.
   * */
  Ts variablesLike(const Ts &likeThese);

  /**
   * \return All tensors in this sub-graph.
   * */
  Ts tensors() { return T::tensors(graph().tensorIds(id()), graph()); }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
