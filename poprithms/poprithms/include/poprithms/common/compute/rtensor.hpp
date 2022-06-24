// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RTENSOR_HPP
#define POPRITHMS_COMMON_COMPUTE_RTENSOR_HPP

#include <memory>
#include <vector>

#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/replication.hpp>
#include <poprithms/common/compute/rsubgraph.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Graph;

using common::compute::DeviceType;
using common::compute::DeviceTypes;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OutIndex;
using common::multiout::OutIndices;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;
using ndarray::DeviceId;
using ndarray::DeviceIds;
using ndarray::DType;
using ndarray::DTypes;
using ndarray::Shape;
using ndarray::Shapes;
using ndarray::TensorInfo;
using ndarray::TensorInfos;
using poprithms::compute::host::CommutativeOp;
using poprithms::ndarray::Dimension;
using poprithms::ndarray::Dimensions;
using poprithms::util::Permutation;
using program::callstack::CallEvent;

/**
 * A tensor is a thin wrapper around (1) a tensor id and (2) a graph, which
 * allows for tensor-centric code where methods are called on tensors instead
 * of on graphs.
 *
 * The suffix '_' rule for tensors:
 *
 * > A method with a trailing '_' returns a tensor which aliases itself.
 *
 * For example in the code,
 *
 * <code>
 * auto b = x.foo_(...);
 * </code>
 *
 * #b is an alias of #x. This notation rule is copied from PyTorch.
 *
 * \param T the tensor class. The RTensor class is templatized to allow users
 *          to create custom tensor classes which inherit from the base tensor
 *          class of this project.
 * */
template <class T> class RTensor {

public:
  /**
   * This tensor is in the callee sub-graph of #cse, and is copied in to.
   *
   * \return The source of the copy to this callee tensor. The source is in
   *         the calling op's sub-graph.
   * */
  T srcInCaller(const CallEvent &cse) const;

  TensorId id() const { return id_; }

  operator TensorId() const { return id(); }

  /**
   * Create a reference to this tensor in the sub-graph \a subGraphId. This
   * allows this tensor to be used in sub-graph #subGraphId.
   *
   * \param subGraphId The SubGraphId of the sub-graph to which the returned
   *                   tensor belongs.
   * */
  T refTo_(SubGraphId subGraphId) const;

  /**
   * Note: an output from a callee cannot be copied to multiple tensors in
   * the calling sub-graph. This rule simplifies the implementation. If you
   * require a tensor to be copied to multiple output indices, insert copies
   * in the calling op after the copy out of the callee.
   *
   * This tensor is in the callee graph of #ce, and is copied to one output in
   * the sub-graph which #callOp is in. This method returns the destination of
   * this copy.
   *
   * \sa Graph::dstInCaller.
   * */
  T dstInCaller(const CallEvent &ce) const;

  /**
   * \return The (shape, dtype, deviceid) triplet of this tensor.
   * */
  TensorInfo info() const;

  RSubGraph<T> subGraph() const;

  /**
   * \return The op which tensor is an output of.
   * */
  const Op &op() const { return graph().computeOp(opId()); }

  /**
   * \return The id of the op which this tensor is an output of.
   * */
  OpId opId() const { return id().opId(); }

  /**
   * \return The shape of this tensor.
   * */
  const Shape &shape() const { return op().outShape(outIndex()); }

  /**
   * \return The output index which this tensor's op outputs this tensor.
   * */
  OutIndex outIndex() const { return id().outIndex(); }

  /**
   * \return The id of the sub-graph to which this tensor belongs.
   * */
  SubGraphId subGraphId() const { return graph().subGraphId(id()); }

  /**
   * \return The total number of elements in this tensor.
   * */
  uint64_t nelms_u64() const { return graph().nelms_u64(id()); }

  /**
   * \return The id of the (unique) device that this tensor belongs to.
   * */
  DeviceId deviceId() const { return graph().deviceId(id()); }

  /**
   * \return The type of the device this tensor is on.
   * */
  DeviceType deviceType() const { return graph().deviceType(deviceId()); }

  /**
   * \return true if this tensor is on an ipu.
   * */
  bool isIpuTensor() const { return deviceType() == DeviceType::Ipu; }

  /**
   * \return The size of the tensor in the dimension #i.
   * */
  int64_t dim(uint64_t i) const { return shape().dim(i); }
  uint64_t dim_u64(uint64_t i) const { return shape().dim_u64(i); }

  /**
   * \return The number of dimensions n this tensor.
   * */
  uint64_t rank_u64() const { return shape().rank_u64(); }
  uint64_t rank_i64() const { return shape().rank_i64(); }

  /**
   * \return The numerical type of the elements of this tensor.
   * */
  DType dtype() const { return graph().dtype(id()); }

  /**
   * \sa Graph::dstInCaller.
   * */
  T dstInCaller(OpId call) const;

  bool graphIsSet() const { return pGraph_; }

  /**
   * \return An alias of this tensor with shape #s. The number of elements of
   *         #s must be the same as the number of elements of this tensor.
   * */
  T reshape_(const Shape &s) const;

  /**
   * \return This tensor, copied and reshaped to have shape #s.
   * */
  T reshape(const Shape &s) const { return reshape_(s).copy(); }

  /**
   * Utility tensor reshaping operations. As usual, the methods which have
   * suffix '_' are aliasing. See the equivalent ndarray::Shape methods for
   * more information on the shapes of the resulting tensors.
   * */
  T flatten() const { return reshape(shape().flatten()); }
  T flatten_() const { return reshape_(shape().flatten()); }
  T squeeze() const { return reshape(shape().squeeze()); }
  T squeeze_() const { return reshape_(shape().squeeze()); }
  T squeeze(const std::vector<uint64_t> &dims) const;
  T squeeze_(const std::vector<uint64_t> &dims) const;
  T unsqueeze(uint64_t d) const { return reshape(shape().unsqueeze(d)); }
  T unsqueeze_(uint64_t d) const { return reshape_(shape().unsqueeze(d)); }
  T flattenTo2d(uint64_t d) const { return reshape(shape().flattenTo2d(d)); }
  T flattenTo2d_(uint64_t d) const {
    return reshape_(shape().flattenTo2d(d));
  }

  /**
   * \return An alias of this tensor. The returned tensor has the same rank as
   *         this tensor, but with the dimensions of this tensor permuted by
   *         #permutation.
   * */
  T dimShuffle_(const Permutation &permutation) const;

  /**
   * \return A copy of this tensor, with its dimensions shuffled.
   * */
  T dimShuffle(const Permutation &p) const { return dimShuffle_(p).copy(); }

  /**
   * Utility tensor dimension shuffling methods. As usual, we use the suffix
   * '_' to denote aliasing tensor methods. See the Permutation class for more
   * information on which dimensions are shuffled with each of these methods.
   * */
  T dimShuffleFinalTwo() const;
  T dimShuffleFinalTwo_() const;
  T dimRoll(uint64_t from, uint64_t to) const;
  T dimRoll_(uint64_t from, uint64_t to) const;

  /**
   * \return An alias of tensor (that it is an alias is implied by the '_'
   *         suffix in the method name). The returned tensor has the same
   *         shape as this tensor, but the dimensions #revDims are all
   *         reversed. All repeated dimensions in #revDims are (effectively)
   *         applied for each repetition, so that revDims=(0,1,0) is
   *         equivalent to revDims=(1) as reversing in dimension 0 twice is
   *         equivalent to not reversing in dimension 0 at all.
   *
   * */
  T reverse_(const Dimensions &revDims) const;

  /**
   * Reverse this tensor along the dimension #d.
   * */
  T reverse_(uint64_t d) const { return reverse_(Dimensions({d})); }

  /**
   * Reverse this tensor, and copy it. See the equivalent aliasing '_' methods
   * for more information.
   * */
  T reverse(uint64_t d) const;
  T reverse(const Dimensions &dims) const;

  /**
   * Broadcast this tensor along the dimensions necessary to create a tensor
   * of shape #expandedShape.
   *
   * \param expandedShape The shape of the expanded view of this tensor.
   *                      #expandedShape must numpy-dominate the shape of this
   *                      tensor. For example this tensor might have shape
   *                      (4,1) and #expandedShape might be (3,4,5).
   * */
  T expand_(const Shape &expandedShape) const;

  T expand(const Shape &s) const { return expand_(s).copy(); }

  /**
   * Expand this tensor along a single dimension #dimension. The dimension
   * #dimensions of this tensor must be a singleton, and it is expanded to
   * size #N.
   * */
  T broadcast_(int64_t N, uint64_t dimension) const {
    return expand_(shape().broadcast(N, dimension));
  }

  /**
   * Concatenate the tensors #ts along dimension #axis.
   * */
  static T concat_(const std::vector<T> &ts, uint64_t axis);

  /**
   * \return A scalar Tensor, whose value is the reduction of all elements
   *         in the Tensor using the commutative op #cop.
   */

  /**
   * Tensor slicing operations. As usual, the suffix '_' denotes aliasing
   * view-changes.
   *
   * \param lower The lower bounds of the slice.
   *
   * \param upper The upper bounds of the slice.
   *
   * #lower and #upper must be of the same rank as this tensor. The resulting
   * slice will have shape '#upper - #lower'. For each dimension of this
   * dimension #d, lower[d] < dim(d) <= upper[d].
   * */
  T slice(const Lower &, const Upper &) const;
  T slice_(const Lower &, const Upper &) const;

  /**
   * Slice in a single dimension #d.
   * */
  T slice(Dimension d, int64_t lower, int64_t upper) const;
  T slice_(Dimension d, int64_t lower, int64_t upper) const;

  /**
   * Slice along a subset of dimensions, #dims.
   * */
  T slice(const Dimensions &dims,
          const std::vector<uint64_t> &lower,
          const std::vector<uint64_t> &upper) const;

  T slice_(const Dimensions &dims,
           const std::vector<uint64_t> &,
           const std::vector<uint64_t> &) const;

  /**
   * Slice this tensor in dimension 0, returning a tensor which is 1 rank
   * lower than this tensor.
   * */
  T at(int64_t d) const {
    return slice(Dimension(0), d, d + 1).squeeze_({0});
  }
  T at_(int64_t d) const {
    return slice_(Dimension(0), d, d + 1).squeeze_({0});
  }

  /**
   * Pad this tensor with a constant, broadcast zero. Example.
   *
   * Suppose that this tensor is 2x3 with values
   *
   *  [[1 2 3]
   *   [4 5 6]]
   *
   * and suppose lower is (0,1) and upper is (0,0). Then the resulting tensor
   * has values
   *
   * [[0 1 2 3]
   *  [0 4 5 6]].
   *  */
  T padWithBroadcastConstZero_(const Lower &lower, const Upper &upper) const;

  /**
   * Copy this tensor to the device #deviceId, which should be a device of the
   * same type as this tensor's.
   * */
  T copy(DeviceId deviceId) const;

  /**
   * Create a copy of this tensor on the same device.
   * */
  T copy() const { return copy(deviceId()); }
  T identity() const { return copy(); }

  /**
   * Reduce this tensor to a rank-0 tensor (a scalar) by using the reduction
   * operation #cop.
   * */
  T reduce(CommutativeOp cop) const;

  /**
   * \return A Tensor of the same rank as this tensor, but reduced to size 1
   *         along Dimensions \a dims.
   * */
  T reduce(const Dimensions &dims, CommutativeOp) const;

  /**
   * \return A tensor of the same rank as this tensor, but reduced to size 1
   *         in Dimension #d.
   * */
  T reduce(Dimension d, CommutativeOp) const;

  /**
   * \return The reduction of this tensor, with shape #out. The shape of this
   *         tensor (s) must satisfy s.numpyBinary(out) = s. See the Shape
   *         class for details.
   * */
  T reduce(const Shape &out, CommutativeOp) const;

  /**
   * Sum-reduce this tensor.
   * */
  T reduceSum() const { return reduce(CommutativeOp::Sum); }
  T reduceSum(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Sum);
  }
  T reduceSum(Dimension d) const { return reduce(d, CommutativeOp::Sum); }
  T reduceSum(const Shape &s) const { return reduce(s, CommutativeOp::Sum); }

  /**
   * Min-reduce this tensor.
   * */
  T reduceMin() const { return reduce(CommutativeOp::Min); }
  T reduceMin(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Min);
  }
  T reduceMin(Dimension d) const { return reduce(d, CommutativeOp::Min); }
  T reduceMin(const Shape &s) const { return reduce(s, CommutativeOp::Min); }

  /**
   * Max-reduce this tensor.
   * */
  T reduceMax() const { return reduce(CommutativeOp::Max); }
  T reduceMax(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Max);
  }
  T reduceMax(Dimension d) const { return reduce(d, CommutativeOp::Max); }
  T reduceMax(const Shape &s) const { return reduce(s, CommutativeOp::Max); }

  /**
   * Product-reduce this tensor.
   * */
  T reduceProduct() const { return reduce(CommutativeOp::Product); }
  T reduceProduct(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Product);
  }
  T reduceProduct(Dimension d) const {
    return reduce(d, CommutativeOp::Product);
  }
  T reduceProduct(const Shape &s) const {
    return reduce(s, CommutativeOp::Product);
  }

  /**
   * Create a tensor for the tensor id #id in the graph #g. If the graph #g
   * does not have a tensor with id #id, the behaviour is undefined.
   * */
  RTensor(const TensorId &, Graph *g);

  Graph &graph() const { return *pGraph_; }

  /**
   * Methods to generate tensors which are the same as this tensor except for
   * one or several of value, device, & subgraph.
   *
   * These methods are similar to the PyTorch "new" method:
   *   b = a.new(13, 19, 23).long()
   */

  /**
   * Create a constant with the same device and subgraph as this tensor.
   * */
  T constant(const HostTensor &) const;
  T constant(DType d, double v) const;

  /**
   * A constant tensor with the same device, subgraph, and type as this
   * tensor.
   * */
  T constant(double v) const { return constant(dtype(), v); }

  /**
   * This static method creates a constant tensor which is like in #t, but
   * with value #v.
   * */
  static T constantLike(const RTensor<T> &t, double v);

  /**
   * Create a constant tensor with the same device and type as this tensor.
   * */
  T constant(SubGraphId, double) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but on
   * device #devId.
   * */
  T variable(DeviceId devId) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * numerical type #dtype.
   * */
  T variable(DType dtype) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * numerical type #dtype and shape #shape.
   * */
  T variable(DType dtype, const Shape &shape) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but in the
   * sub-graph #sgId.
   * */
  T variable(SubGraphId sgId) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * shape #shape.
   * */
  T variable(const Shape &shape) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * shape #shape and on device #deviceId.
   * */
  T variable(const Shape &shape, DeviceId deviceId) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor in every
   * respect.
   * */
  T variable() const;

  /**
   * Binary elementwise operations using numpy broadcasting rules, see
   * https://numpy.org/doc/stable/user/basics.broadcasting.html for more
   * information.
   *
   * As with all other methods, the '_' suffix denotes an inplace operation.
   * This tensor's shape must numpy-dominate the second argument's, and the
   * output has the shape as this tensor.
   * */
  T add(const RTensor<T> &) const;
  T add_(const RTensor<T> &) const;

  /**
   * Elementwise multiply this tensor with #rhs.
   * */
  T mul(const RTensor<T> &rhs) const;
  T mul_(const RTensor<T> &rhs) const;

  /**
   * Subtract #rhs from this tensor.
   * */
  T sub(const RTensor<T> &rhs) const;
  T sub_(const RTensor<T> &rhs) const;

  /**
   * Divide this tensor by #rhs.
   * */
  T div(const RTensor<T> &rhs) const;
  T div_(const RTensor<T> &rhs) const;

  /**
   * \return This tensor to the power of #rhs.
   * */
  T pow(const RTensor<T> &rhs) const;
  T pow_(const RTensor<T> &rhs) const;

  /**
   * Copy the values in #rhs to this tensor. This tensor operation supports
   * numpy broadcasting, so #rhs must have a shape which is numpy
   * broadcastable to this tensor's shape.
   * */
  T copyFrom_(const RTensor<T> &rhs) const;
  T update_(const RTensor<T> &rhs) const { return copyFrom_(rhs); }

  /**
   * \return A boolean tensor which is true where this tensor is greater than
   *         #rhs. This tensor operation supports numpy broadcasting.
   * */
  T greaterThan(const RTensor<T> &rhs) const;

  /**
   * The natural logarithm of this tensor.
   * */
  T log_() const;
  T log() const;

protected:
  /**
   * Create an op of type TOp in this tensor's graph. The new op will have
   * inputs #inIds, and the outputs will have (shape, dtype, deviceId)
   * provided by #outInfos. Additional op attributes are #opAtts.
   * */
  template <class TOp, class... Args>
  OpId createComputeOp(const TensorIds &inIds,
                       const TensorInfos &outInfos,
                       Args &&...opAtts) const;

  template <class TOp, class... Args>
  T createTensor(const TensorIds &ins,
                 const TensorInfos &outs,
                 Args &&...args) const {
    return {{createComputeOp<TOp>(ins, outs, std::forward<Args>(args)...), 0},
            &graph()};
  }

  template <class TOp, class... Args>
  T createUnaryWithSameInfo(Args &&...args) const {
    return createTensor<TOp>({id()}, {info()}, std::forward<Args>(args)...);
  }

  template <class TOp, class... Args>
  T createWithNumpyShape(const TensorIds &ins, Args &&...args) const {
    return createTensor<TOp>(
        ins,
        {info().withShape(Shape::numpyVariadic(graph().shapes(ins)))},
        std::forward<Args>(args)...);
  }

  /**
   * Create a tensor by applying a one-to-one view-change op of type TOp to
   * this tensor. The created tensor has shape #outShape. If the view-change
   * is effectively the identity view-change, then no new op is created in the
   * graph, and this tensor is returned directly (so that the returned tensor
   * has same id as this tensor).
   * */
  template <class TOp, class... Args>
  T createUnaryViewChange(const Shape &outShape, Args &&...args) const {

    // If the view-change is an identity view, do not create a new tensor --
    // just return this tensor.
    if (TOp::isIdentity(shape(), outShape, args...)) {
      return T(id(), &graph());
    }
    return createTensor<TOp>(
        {id()}, {info().withShape(outShape)}, std::forward<Args>(args)...);
  }

  template <class TOp, class... Args>
  T createUnaryWithNewShape(const Shape &s, Args &&...args) const {
    return createTensor<TOp>(
        {id()}, {info().withShape(s)}, std::forward<Args>(args)...);
  }

private:
  TensorId id_;
  Graph *pGraph_;
};

template <typename T> T operator*(const RTensor<T> &a, const RTensor<T> &b) {
  return a.mul(b);
}

template <typename T> T operator+(const RTensor<T> &a, const RTensor<T> &b) {
  return a.add(b);
}

template <typename T> T operator/(const RTensor<T> &a, const RTensor<T> &b) {
  return a.div(b);
}

template <typename T> T operator-(const RTensor<T> &a, const RTensor<T> &b) {
  return a.sub(b);
}

template <typename T> T concat_(const std::vector<T> &ts, uint64_t axis) {
  return RTensor<T>::concat_(ts, axis);
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
