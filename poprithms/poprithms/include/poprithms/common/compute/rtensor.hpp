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
#include <poprithms/ndarray/unfold.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Graph;

class MatMulOptions {
  // Currently there are no options for matmul, this is a just placeholder for
  // the future.
public:
  bool operator==(const MatMulOptions &) const { return true; }
};

/**
 * Options for ops which copy between host and ipu.
 * */
class CopyBetweenHostAndIpuOptions {

public:
  CopyBetweenHostAndIpuOptions()  = default;
  ~CopyBetweenHostAndIpuOptions() = default;

  uint64_t bufferingDepth() const { return bufferingDepth_; }
  CopyBetweenHostAndIpuOptions &bufferingDepth(uint64_t v) {
    bufferingDepth_ = v;
    return *this;
  }

  bool operator==(const CopyBetweenHostAndIpuOptions &rhs) const {
    return t() == rhs.t();
  }

  bool operator<(const CopyBetweenHostAndIpuOptions &rhs) const {
    return t() < rhs.t();
  }

private:
  std::tuple<uint64_t> t() const { return bufferingDepth_; }
  uint64_t bufferingDepth_ = 1ull;
};

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

  using Ts = std::vector<T>;

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
   * Reshape the dimensions in the range [dim0, dim1). See
   * Shape::reshapePartial_ for more information.
   * */
  T reshapePartial_(uint64_t dim0,
                    uint64_t dim1,
                    const std::vector<int64_t> &newDims) {
    return reshape_(shape().reshapePartial(dim0, dim1, newDims));
  }

  /**
   * Flatten the dimensions in range [dim0, dim1) into a single dimension.
   * */
  T flatten_(uint64_t dim0, uint64_t dim1) const {
    return reshape_(shape().flatten(dim0, dim1));
  }

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
   * The method #at (above) is static. Specifically, the slice in dimension-0
   * that #at takes is always the same [d, d+1) where #d is known at compile
   * time.
   *
   * This #dynamicAt method is a dynamic equivalent of #at. The shape of the
   * output is exactly the same as for #at. Specifically, if this tensor has
   * shape (s0,s1,..,sZ) then the returned tensor has shape (s1,...,sZ). But
   * instead of always taking the slice [d,d+1) for a fixed #d, the specific
   * slice taken is a runtime variable.
   *
   * The tensor #d must an unsigned integer and be in the range [0, dim(0)).
   * */
  T dynamicAt(const RTensor<T> &d) const {
    return dynamicMultiSlice(d.reshape_({1, 1}), Dimensions({0}), {1})
        .squeeze_({0, 1});
  }

  /**
   * Upsample this tensor in dimension #d. Example:
   *
   * If this tensor has values (1,2,3) and N=2 and dim=0, the returned tensor
   * has values (1,1,2,2,3,3).
   * */
  T upsample_(uint64_t N, Dimension dim) const {
    auto d = dim.get();
    return unsqueeze_(d + 1).broadcast_(N, d + 1).flatten_(d, d + 2);
  }

  /**
   * Unfold this tensor in dimension #d, with stride #step and slices of size
   * #size. See ndarray::Unfolder for more information.
   * */
  T unfold_(Dimension d, uint64_t size, uint64_t step) const {
    using H = ndarray::TUnfoldHelper<T>;
    return ndarray::Unfolder<T, H>::unfold(
        T(id(), &graph()), d.get(), size, step);
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
  T variable(DeviceId deviceId, const Shape &shape) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but on
   * device #deviceId and in sub-graph #sgId.
   * */
  T variable(DeviceId deviceId, SubGraphId sgId) const;

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
   * Add this tensor to a scalar of the same type, of value #v.
   * */
  T add_(double v) const { return add_(constant(v)); }
  T add(double v) const { return add(constant(v)); }

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

  T pow(double v) const { return pow(constant(v)); }
  T pow_(double v) const { return pow_(constant(v)); }

  T mul(double v) const { return mul(constant(v)); }
  T mul_(double v) const { return mul_(constant(v)); }

  T div(double v) const { return div(constant(v)); }
  T div_(double v) const { return div_(constant(v)); }

  T sub(double v) const { return sub(constant(v)); }
  T sub_(double v) const { return sub_(constant(v)); }

  /**
   * \return The remainder when this tensor is divided by #rhs. #rhs must have
   *         the same dtype as this tensor.
   *
   * This elementwise operation is identical to fmod for floating point
   * numbers, as defined for C++.
   * */
  T rem(const RTensor<T> &rhs) const;
  T rem_(const RTensor<T> &rhs) const;

  T modulo_(uint64_t v) const { return rem_(constant(v)); }
  T modulo(uint64_t v) const { return rem(constant(v)); }

  T tickModulo_(uint64_t m) const { return add_(1).modulo_(m); }

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
   * \return A boolean tensor which is true where this tensor is (bitwise)
   *         equal to #rhs. This tensor operation supports numpy broadcasting.
   * */
  T equalTo(const RTensor<T> &rhs) const;

  /**
   * Matrix multiply, using numpy broadcasting rules, see
   * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
   *
   * \param arg1 the right-hand side argument of the matrix multiply.
   * */
  T matmul(const RTensor<T> &rhs, DType outType, const MatMulOptions &) const;

  T matmul(const RTensor<T> &arg1) const {
    return matmul(arg1, arg1.dtype(), {});
  }

  /**
   * Encode this tensor with 0's and 1's. This tensor must be of shape (N, C),
   * with #indices of shape (N,). The elements of indices must all be in the
   * range 0 <= v < C. #indices must be of an integral type. This tensor is
   * encoded, inplace, with a single 1 per row, the specific row defined by
   * #indices. Specifically if we call this tensor 't', then t[r][indices[r]]
   * is encoded with value 1.
   * */
  T encodeOneHot01_(const RTensor<T> &indices) const;

  /**
   * Similar to encodeOneHot01_ but instead of '0' the value of the scalar
   * tensor 'off' is used, and instead of '1' the value of the scalar tensor
   * 'on' is used. #off and #on must have the same numerical type as this
   * tensor.
   * */
  T encodeOneHotOffOn_(const RTensor<T> &indices,
                       const RTensor<T> &off,
                       const RTensor<T> &on) const;

  /**
   * The natural logarithm of this tensor.
   * */
  T log_() const;
  T log() const;

  /**
   * Negate all elements of this tensor.
   * */
  T neg_() const;
  T neg() const;

  /**
   * Cast this tensor to a tensor of type #outType.
   * */
  T to(DType outType) const;

  /** Unary elementwise operations */
  T zero_() const { return fill_(HostTensor::scalar(dtype(), 0)); }

  /**
   * The absolute value of this tensor.
   * */
  T abs_() const;
  T abs() const;

  /**
   * The sine of this tensor.
   * */
  T sin_() const;
  T sin() const;

  /**
   * The cosine of this tensor.
   * */
  T cos_() const;
  T cos() const;

  /**
   * The sign, or signum, of this tensor:
   *
   *             = -1 if x < 0
   * signum(x)   =  0 if x = 0
   *             = +1 if x > 0.
   *
   * The returned tensor has the same dtype as this tensor.
   **/
  T signum_() const;
  T signum() const;

  /**
   * The square root of this tensor.
   * */
  T sqrt_() const;
  T sqrt() const;

  /**
   * e^(this tensor) where e is the transcendental value 2.718...
   * */
  T exp_() const;
  T exp() const;

  /**
   * The rectified linear unit of this tensor.
   * */
  T relu_() const { return mul_(greaterThan(constant(0)).to(dtype())); }
  T relu() const { return mul(greaterThan(constant(0)).to(dtype())); }

  /**
   * Fill this tensor with the scalar value #vScalar.
   * */
  T fill_(const HostTensor &vScalar) const;
  T setToZero_() const { return fill_(HostTensor::scalar(dtype(), 0)); }

  T name(const std::string &n) const {
    graph().setName(id(), n);
    return T(id(), &graph());
  }

  static std::vector<T> tensors(const TensorIds &ids, Graph &g);

  /**
   * Update this ipu tensor by copying to it from the host tensor
   * #sourceOnHost. The returned tensor is an alias of this ipu tensor.
   *
   * If this tensor has shape #s, then #sourceOnHost tensor must have shape
   * (cbc, rf, *s) where:
   *
   * #cbc is the size of the circular buffer of the host tensor. Subsequent
   *      calls to this method will copy from subsequent slices of
   *      #sourceFromHost in dimension 0. When this method has been called
   *      #cbc times, the copy source index returns to zero (that is why it is
   *      a 'circular' buffer).
   *
   * #rf is either:
   *    (1) the replication factor of this tensor, or
   *    (2) 1. In this case, the host value is broadcast to all replicas.
   *
   * To support something  inbetween (1) and (2), where host tensors are
   * broadcast to subsets of all replicas, (1) must be used. That is, the
   * broadcasting must be done on host, beforehand.
   * */
  T updateFromHost_(const RTensor<T> &sourceOnHost,
                    const CopyBetweenHostAndIpuOptions &opts = {}) const;

  /**
   * Copy this host tensor to ipu. \sa updateFromHost_.
   * */
  T hostToIpu(DeviceId ipuDestination,
              const CopyBetweenHostAndIpuOptions & = {}) const;

  /**
   * Update this host tensor by copying to it from an ipu tensor.
   *
   * If #sourceOnIpu is of shape #s, then this tensor must of of shape (#cbc,
   * #rf, *s), where #cbc is the circular buffer count of this host tensor and
   * #rf is the replication factor of the graph. Subsequent calls to this
   * method will write to subsequent slices in dimension 0 of the host tensor.
   *
   * Copying (1) from ipu to host and (2) from host to ipu, have identical
   * shape requirements. \sa updateFromHost_ for more information.
   * */
  T updateFromIpu_(const RTensor<T> &sourceOnIpu,
                   const CopyBetweenHostAndIpuOptions &opts = {}) const;

  /**
   * Copy this ipu tensor to host. \sa updateFromIpu_.
   * */
  T ipuToHost(CircularBufferCount,
              const CopyBetweenHostAndIpuOptions & = {}) const;

  /**
   * \return The root reference tensor of this tensor. For tensors which are
   *         created with a call to #refTo_, this is the tensor on which
   *         #refTo_ was called. For all other tensors, this method returns
   *         the tensor itself. \sa RefFrom_ op.
   * */
  T rootRef() const { return {op().rootRef(outIndex()), &graph()}; }

  /**
   * \sa Graph::refsExcludingSelf.
   * */
  Ts refsExcludingSelf() const {
    return tensors(op().refsExcludingSelf(outIndex()), graph());
  }

  /**
   * The ids of the tensors in #ts.
   * */
  static TensorIds tensorIds(const std::vector<T> &ts);

  bool isRootRef() const { return id() == rootRef().id(); }

  uint64_t nDerivedRefs() const { return op().nDerivedRefs(outIndex()); }

  bool hasDerivedRefs() const { return nDerivedRefs() != 0; }

  void setName(const std::string &nm) const { graph().setName(id(), nm); }

  T inTensor(InIndex i) const {
    return {graph().inTensorId(opId(), i), &graph()};
  }

  /**
   * \return A boolean tensor that is true where this tensor is strictly
   *         positive.
   * */
  T isStrictlyPositive() const { return greaterThan(constant(dtype(), 0.0)); }

  /**
   * This tensor is copied from the calling scope into a callee sub-graph,
   * defined by #ce. This method returns all of the tensors in the callee
   * sub-graph that it is copied to.
   * */
  Ts dstsInCallee(const CallEvent &ce) const {
    return tensors(graph().dstsInCallee(id(), ce), graph());
  }

  /**
   * This tensor is the destination of a copy out of a callee subgraph. This
   * method returns the source of this copy.
   * */
  T srcInCallee(uint64_t calleeIndex) const {
    return {
        graph().srcInCallee({opId(), subGraphId(), calleeIndex}, outIndex()),
        &graph()};
  }

  bool isFixedPoint() const {
    return poprithms::ndarray::isFixedPoint(dtype());
  }

  /**
   * \param offset A rank-2 fixed-point tensor. The first dimension is the
   *               number of slices to take. The second dimensions contains
   *               the offsets in the slice dimensions, for each of the
   *               slices.
   *
   * \param dims The dimensions of this tensor to slice.
   *
   * \param sizes The sizes of the slices in the dimensions #dims.
   *
   * Shape expectations:
   *
   *  this tensor : (D0, D1, D2, ... DZ).
   *  offset      : a tensor of shape (N, K).
   *  dims        : vector with K elements.
   *  sizes       : vector with K elements.
   *
   *  returned tensor : (N, D0', D1', ... DZ')
   *
   *  where Dj' = Dj if j is not in dims.
   *            = D'[sizes[k]] where dims[k] = j (<= Dj).
   *
   * Example:
   * - this tensor has shape (3,4,5)
   * - offset has shape (10,2)
   * - dims is (0,2)
   * - sizes is (2,3)
   * ==> the returned tensors has shape (10,2,4,3):
   *
   *      0 1 2
   *      |   |
   *      v   v  dims.
   *     (3,4,5)
   *      |   |
   *      v   v
   *     (2,4,3) replace sizes in dims with sizes.
   *   |
   *   v
   *  (10,2,4,3) prepend with the number of shapes.
   *
   * The start:end of the slices is dynamic, although the size (end - start)
   * is static. The start values are contained in the #offset tensor.
   * */
  T dynamicMultiSlice(const RTensor<T> &offset,
                      const Dimensions &dims,
                      const Shape &sizes) const;

  /**
   * This tensor is the slice tensor, and is updated inplace with sliceable.
   * It has the same shape as the output of dynamicMultiSlice.
   * */
  T dynamicMultiSlice_(const RTensor<T> &sliceable,
                       const RTensor<T> &offsets,
                       const Dimensions &) const;

  /**
   * Similar to dynamicMultiSlice, but only 1 slice is taken. The output has
   * the same rank as this tensor. #offset is a a rank-1 tensor.
   * */
  T dynamicSlice(const RTensor<T> &offset,
                 const Dimensions &dims,
                 const Shape &shape) const {
    return dynamicMultiSlice(offset.unsqueeze_(0), dims, shape).squeeze_({0});
  }

  /**
   * This method is very similar to #dynamicMultiSlice_, but the copy happens
   * in the opposite direction. Whereas #dynamicMultiSlice_ does a copy
   * from the sliceable ('low-rank and wide') tensor to the slice ('high-rank
   * and narrow') tensor, this method does a copy from a 'slice' tensor to a
   * 'sliceable' tensor.
   *
   * \param slice The source of copy. This tensor has a rank which is 1
   *              higher than this tensor.
   *
   * \param offset A rank-2 tensor, where the first dimension is the number of
   *              slices.
   *
   * \sa dynamicMultiSlice_.
   * */
  T dynamicMultiUpdate_(const RTensor<T> &slice,
                        const RTensor<T> &offset,
                        const Dimensions &) const;

  /**
   * \sa dynamicMultiUpdate_
   *
   * \param slice A tensor which has the same rank as this tensor, and is
   *              smaller that this tensor in the dimensions #dims.
   *
   * \param offset A rank-1 tensor, of the same size as #dims.
   * */
  T dynamicUpdate_(const RTensor<T> &slice,
                   const RTensor<T> &offset,
                   const Dimensions &dims) const {
    return dynamicMultiUpdate_(
        slice.unsqueeze_(0), offset.unsqueeze_(0), dims);
  }

  /**
   *
   * Update a slice in dimension-0 of this tensor.
   *
   * \param index a rank-0 scalar tensor. Values are in range [0, dim(0)).
   *
   * \param slice a tensor whose shape is this tensor's shape from
   *              dimension 1 onwards.
   *
   * Example: If this tensor has shape (5,4,3) then #slice must have
   * shape (4,3) and #index is a scalar in the range [0,5).
   * */

  T updateAt_(const RTensor<T> &slice, const RTensor<T> &index) const {
    return dynamicMultiUpdate_(slice.reshape_(slice.shape().prependOnes(2)),
                               index.reshape_({1, 1}),
                               Dimensions{0});
  }

  /**
   * The inverse operation of pushToStash.
   * */
  T popFromStash(const RTensor<T> &index) { return dynamicAt(index); }

  /**
   * This 'sliceable' tensor must be of rank-2, of shape (M, S). It is updated
   * inplace with maximum values from the 'slice' tensor #source.
   *
   * \param source is a tensor of shape (N, S). N can be thought of as a
   *               'dictionary' size, and S can be thought of as the size of
   *               words in the dictionary.
   *
   * \param offsets is of shape (N) where the elements are fixed-point values
   *                in the range [0,S).
   *
   * Example with M=4 S=2 N=3:
   *   this tensor is   source is     offsets is
   *        1 2           10 12             1
   *        3 4           11 0              2
   *        5 6           9  20             1
   *        7 8
   *
   *    then the udpated tensor is
   *        1  2
   *        10 20
   *        11 6
   *        7  8.
   *
   * This op is the same as PyTorch's scatter-max:
   * https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/max.html
   * */
  T dynamicMultiUpdateMax_(const RTensor<T> &source,
                           const RTensor<T> &offsets) const;

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

  template <class TOp, class... Args>
  T createBooleanWithNumpyShape(const TensorIds &ins, Args &&...args) const {
    return createTensor<TOp>(
        ins,
        {info()
             .withShape(Shape::numpyVariadic(graph().shapes(ins)))
             .withDType(DType::Boolean)},
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

  const Op &op(OpId opId) const { return graph().computeOp(opId); }
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

template <typename T> T matmul(const RTensor<T> &t0, const RTensor<T> &t1) {
  return t0.matmul(t1);
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
