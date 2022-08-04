// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TENSOR_HPP
#define POPRITHMS_COMMON_COMPUTE_TENSOR_HPP

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/common/compute/copyoptions.hpp>
#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/compute/matmuloptions.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/remoteoptions.hpp>
#include <poprithms/common/compute/replication.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/multiout/toptionaltensor.hpp>
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
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::compute::host::CommutativeOp;
using poprithms::ndarray::Dimension;
using poprithms::ndarray::Dimensions;
using poprithms::ndarray::Offsets;
using poprithms::util::Permutation;
using program::callstack::CallEvent;

struct NllGrad;
class Tensor;
class SubGraph;

using Tensors         = std::vector<Tensor>;
using OptionalTensor  = poprithms::common::multiout::TOptionalTensor<Tensor>;
using OptionalTensors = std::vector<OptionalTensor>;

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
 * */
class Tensor {

public:
  /**
   * Create a tensor for the tensor id #id in the graph #g. If the graph #g
   * does not have a tensor with id #id, the behaviour is undefined.
   * */
  Tensor(const TensorId &, Graph *g);

  Tensor() = delete;

  TensorId id() const { return id_; }

  operator TensorId() const { return id(); }

  /**
   * This tensor is in the callee sub-graph of #cse, and is copied in to.
   *
   * \return The source of the copy to this callee tensor. The source is in
   *         the calling op's sub-graph.
   * */
  Tensor srcInCaller(const CallEvent &cse) const;

  /**
   * Create a reference to this tensor in the sub-graph \a subGraphId. This
   * allows this tensor to be used in sub-graph #subGraphId.
   *
   * \param subGraphId The SubGraphId of the sub-graph to which the returned
   *                   tensor belongs.
   * */
  Tensor refTo_(SubGraphId subGraphId) const;

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
  Tensor dstInCaller(const CallEvent &ce) const;

  /**
   * \return The (shape, dtype, deviceid) triplet of this tensor.
   * */
  TensorInfo info() const;

  /**
   * \return The op which tensor is an output of.
   * */
  const Op &op() const;

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
  SubGraphId subGraphId() const;

  /**
   * \return The total number of elements in this tensor.
   * */
  uint64_t nelms_u64() const;

  /**
   * \return The id of the (unique) device that this tensor belongs to.
   * */
  DeviceId deviceId() const;

  /**
   * \return The type of the device this tensor is on.
   * */
  DeviceType deviceType() const;

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
  DType dtype() const;

  /**
   * \sa Graph::dstInCaller.
   * */
  Tensor dstInCaller(OpId call) const;

  bool graphIsSet() const { return pGraph_; }

  /**
   * \return An alias of this tensor with shape #s. The number of elements of
   *         #s must be the same as the number of elements of this tensor.
   * */
  Tensor reshape_(const Shape &s) const;

  /**
   * Reshape the dimensions in the range [dim0, dim1). See
   * Shape::reshapePartial_ for more information.
   * */
  Tensor reshapePartial_(uint64_t dim0,
                         uint64_t dim1,
                         const std::vector<int64_t> &newDims) {
    return reshape_(shape().reshapePartial(dim0, dim1, newDims));
  }

  /**
   * Flatten the dimensions in range [dim0, dim1) into a single dimension.
   * */
  Tensor flatten_(uint64_t dim0, uint64_t dim1) const {
    return reshape_(shape().flatten(dim0, dim1));
  }

  /**
   * \return This tensor, copied and reshaped to have shape #s.
   * */
  Tensor reshape(const Shape &s) const { return reshape_(s).copy(); }

  /**
   * Utility tensor reshaping operations. As usual, the methods which have
   * suffix '_' are aliasing. See the equivalent ndarray::Shape methods for
   * more information on the shapes of the resulting tensors.
   * */
  Tensor flatten() const { return reshape(shape().flatten()); }
  Tensor flatten_() const { return reshape_(shape().flatten()); }

  Tensor squeeze() const { return reshape(shape().squeeze()); }
  Tensor squeeze_() const { return reshape_(shape().squeeze()); }
  Tensor squeeze(const std::vector<uint64_t> &dims) const;
  Tensor squeeze_(const std::vector<uint64_t> &dims) const;
  Tensor unsqueeze(uint64_t d) const { return reshape(shape().unsqueeze(d)); }
  Tensor unsqueeze_(uint64_t d) const {
    return reshape_(shape().unsqueeze(d));
  }
  Tensor flattenTo2d(uint64_t d) const {
    return reshape(shape().flattenTo2d(d));
  }
  Tensor flattenTo2d_(uint64_t d) const {
    return reshape_(shape().flattenTo2d(d));
  }

  /**
   * \return An alias of this tensor. The returned tensor has the same rank as
   *         this tensor, but with the dimensions of this tensor permuted by
   *         #permutation.
   * */
  Tensor dimShuffle_(const Permutation &permutation) const;

  /**
   * \return A copy of this tensor, with its dimensions shuffled.
   * */
  Tensor dimShuffle(const Permutation &p) const {
    return dimShuffle_(p).copy();
  }

  /**
   * Utility tensor dimension shuffling methods. As usual, we use the suffix
   * '_' to denote aliasing tensor methods. See the Permutation class for more
   * information on which dimensions are shuffled with each of these methods.
   * */
  Tensor dimShuffleFinalTwo() const;
  Tensor dimShuffleFinalTwo_() const;
  Tensor dimRoll(uint64_t from, uint64_t to) const;
  Tensor dimRoll_(uint64_t from, uint64_t to) const;

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
  Tensor reverse_(const Dimensions &revDims) const;

  /**
   * Reverse this tensor along the dimension #d.
   * */
  Tensor reverse_(uint64_t d) const { return reverse_(Dimensions({d})); }

  /**
   * Reverse this tensor, and copy it. See the equivalent aliasing '_' methods
   * for more information.
   * */
  Tensor reverse(uint64_t d) const;
  Tensor reverse(const Dimensions &dims) const;

  /**
   * Broadcast this tensor along the dimensions necessary to create a tensor
   * of shape #expandedShape.
   *
   * \param expandedShape The shape of the expanded view of this tensor.
   *                      #expandedShape must numpy-dominate the shape of this
   *                      tensor. For example this tensor might have shape
   *                      (4,1) and #expandedShape might be (3,4,5).
   * */
  Tensor expand_(const Shape &expandedShape) const;

  Tensor expand(const Shape &s) const { return expand_(s).copy(); }

  /**
   * Expand this tensor along a single dimension #dimension. The dimension
   * #dimensions of this tensor must be a singleton, and it is expanded to
   * size #N.
   * */
  Tensor broadcast_(int64_t N, uint64_t dimension) const {
    return expand_(shape().broadcast(N, dimension));
  }

  /**
   * Concatenate the tensors #ts along dimension #axis.
   * */
  static Tensor concat_(const Tensors &ts, uint64_t axis);

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
  Tensor slice(const Lower &, const Upper &) const;
  Tensor slice_(const Lower &, const Upper &) const;

  /**
   * Slice in a single dimension #d.
   * */
  Tensor slice(Dimension d, int64_t lower, int64_t upper) const;
  Tensor slice_(Dimension d, int64_t lower, int64_t upper) const;

  /**
   * Slice along a subset of dimensions, #dims.
   * */
  Tensor slice(const Dimensions &dims,
               const std::vector<uint64_t> &lower,
               const std::vector<uint64_t> &upper) const;

  Tensor slice_(const Dimensions &dims,
                const std::vector<uint64_t> &,
                const std::vector<uint64_t> &) const;

  /**
   * Slice this tensor in dimension 0, returning a tensor which is 1 rank
   * lower than this tensor.
   * */
  Tensor at(int64_t d) const {
    return slice(Dimension(0), d, d + 1).squeeze_({0});
  }
  Tensor at_(int64_t d) const {
    return slice_(Dimension(0), d, d + 1).squeeze_({0});
  }

  /**
   * Update a statically defined region of this tensor. The shape of the
   * region being updated is the same as #update. The start (lower bound) of
   * the region being updated is #offset.
   * */
  Tensor update_(const Tensor &update, const Offsets &) const;

  /**
   * Update a statically defined region of this tensor. This is similar to
   * #dynamicUpdate_, except that the offset is statically defined as opposed
   * to being a tensor.
   * */
  Tensor
  update_(const Tensor &update, const Offsets &, const Dimensions &) const;

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
  Tensor dynamicAt(const Tensor &d) const {
    return dynamicMultiSlice(d.reshape_({1, 1}), Dimensions({0}), {1})
        .squeeze_({0, 1});
  }

  /**
   * Upsample this tensor in dimension #d. Example:
   *
   * If this tensor has values (1,2,3) and N=2 and dim=0, the returned tensor
   * has values (1,1,2,2,3,3).
   * */
  Tensor upsample_(uint64_t N, Dimension dim) const {
    auto d = dim.get();
    return unsqueeze_(d + 1).broadcast_(N, d + 1).flatten_(d, d + 2);
  }

  /**
   * Unfold this tensor in dimension #d, with stride #step and slices of size
   * #size. See ndarray::Unfolder for more information.
   * */
  Tensor unfold_(Dimension d, uint64_t size, uint64_t step) const;

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
  Tensor padWithBroadcastConstZero_(const Lower &lower,
                                    const Upper &upper) const;

  /**
   * Copy this tensor to the device #deviceId, which should be a device of the
   * same type as this tensor's.
   * */
  Tensor copy(DeviceId deviceId) const;

  /**
   * Create a copy of this tensor on the same device.
   * */
  Tensor copy() const { return copy(deviceId()); }
  Tensor identity() const { return copy(); }

  /**
   * Reduce this tensor to a rank-0 tensor (a scalar) by using the reduction
   * operation #cop.
   * */
  Tensor reduce(CommutativeOp cop) const;

  /**
   * \return A Tensor of the same rank as this tensor, but reduced to size 1
   *         along Dimensions \a dims.
   * */
  Tensor reduce(const Dimensions &dims, CommutativeOp) const;

  /**
   * \return A tensor of the same rank as this tensor, but reduced to size 1
   *         in Dimension #d.
   * */
  Tensor reduce(Dimension d, CommutativeOp) const;

  /**
   * \return The reduction of this tensor, with shape #out. The shape of this
   *         tensor (s) must satisfy s.numpyBinary(out) = s. See the Shape
   *         class for details.
   * */
  Tensor reduce(const Shape &out, CommutativeOp) const;

  /**
   * Sum-reduce this tensor.
   * */
  Tensor reduceSum() const { return reduce(CommutativeOp::Sum); }
  Tensor reduceSum(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Sum);
  }
  Tensor reduceSum(Dimension d) const {
    return reduce(d, CommutativeOp::Sum);
  }
  Tensor reduceSum(const Shape &s) const {
    return reduce(s, CommutativeOp::Sum);
  }

  /**
   * Min-reduce this tensor.
   * */
  Tensor reduceMin() const { return reduce(CommutativeOp::Min); }
  Tensor reduceMin(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Min);
  }
  Tensor reduceMin(Dimension d) const {
    return reduce(d, CommutativeOp::Min);
  }
  Tensor reduceMin(const Shape &s) const {
    return reduce(s, CommutativeOp::Min);
  }

  /**
   * Max-reduce this tensor.
   * */
  Tensor reduceMax() const { return reduce(CommutativeOp::Max); }
  Tensor reduceMax(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Max);
  }
  Tensor reduceMax(Dimension d) const {
    return reduce(d, CommutativeOp::Max);
  }
  Tensor reduceMax(const Shape &s) const {
    return reduce(s, CommutativeOp::Max);
  }

  /**
   * Product-reduce this tensor.
   * */
  Tensor reduceProduct() const { return reduce(CommutativeOp::Product); }
  Tensor reduceProduct(const Dimensions &dims) const {
    return reduce(dims, CommutativeOp::Product);
  }
  Tensor reduceProduct(Dimension d) const {
    return reduce(d, CommutativeOp::Product);
  }
  Tensor reduceProduct(const Shape &s) const {
    return reduce(s, CommutativeOp::Product);
  }

  /**
   * This tensor is on an ipu. The returned tensor is on the same device, and
   * has the same shape. Each replica in the returned tensor is the summation
   * of replicas of this tensor. For example if this tensor has shape (2,) and
   * the replication factor is three, and the 3 replicas have values [1,2],
   * [3,4], and [5,-1] respectively, then the result (on all replicas) is
   * [9,5].
   * */
  Tensor reduceSumAcrossReplicas() const;
  Tensor reduceSumAcrossReplicas_() const;

  Graph &graph() const;

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
  Tensor constant(const HostTensor &) const;
  Tensor constant(DType d, double v) const;

  /**
   * A constant tensor with the same device, subgraph, and type as this
   * tensor.
   * */
  Tensor constant(double v) const { return constant(dtype(), v); }

  /**
   * This static method creates a constant tensor which is like in #t, but
   * with value #v.
   * */
  static Tensor constantLike(const Tensor &t, double v);

  /**
   * Create a constant tensor with the same device and type as this tensor.
   * */
  Tensor constant(SubGraphId, double) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but on
   * device #devId.
   * */
  Tensor variable(DeviceId devId) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * numerical type #dtype.
   * */
  Tensor variable(DType dtype) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * numerical type #dtype and shape #shape.
   * */
  Tensor variable(DType dtype, const Shape &shape) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but in the
   * sub-graph #sgId.
   * */
  Tensor variable(SubGraphId sgId) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * shape #shape.
   * */
  Tensor variable(const Shape &shape) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but of
   * shape #shape and on device #deviceId.
   * */
  Tensor variable(DeviceId deviceId, const Shape &shape) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor, but on
   * device #deviceId and in sub-graph #sgId.
   * */
  Tensor variable(DeviceId deviceId, SubGraphId sgId) const;

  /**
   * Create a new variable (non-constant) tensor like this tensor in every
   * respect.
   * */
  Tensor variable() const;

  /**
   * Binary elementwise operations using numpy broadcasting rules, see
   * https://numpy.org/doc/stable/user/basics.broadcasting.html for more
   * information.
   *
   * As with all other methods, the '_' suffix denotes an inplace operation.
   * This tensor's shape must numpy-dominate the second argument's, and the
   * output has the shape as this tensor.
   * */
  Tensor add(const Tensor &) const;
  Tensor add_(const Tensor &) const;

  /**
   * Add this tensor to a scalar of the same type, of value #v.
   * */
  Tensor add_(double v) const { return add_(constant(v)); }
  Tensor add(double v) const { return add(constant(v)); }

  /**
   * Elementwise multiply this tensor with #rhs.
   * */
  Tensor mul(const Tensor &rhs) const;
  Tensor mul_(const Tensor &rhs) const;

  /**
   * Subtract #rhs from this tensor.
   * */
  Tensor sub(const Tensor &rhs) const;
  Tensor sub_(const Tensor &rhs) const;

  /**
   * Divide this tensor by #rhs.
   * */
  Tensor div(const Tensor &rhs) const;
  Tensor div_(const Tensor &rhs) const;

  /**
   * \return This tensor to the power of #rhs.
   * */
  Tensor pow(const Tensor &rhs) const;
  Tensor pow_(const Tensor &rhs) const;

  /**
   * \return The maximum of this tensor and #rhs.
   * */
  Tensor max(const Tensor &rhs) const;
  Tensor max_(const Tensor &rhs) const;

  /**
   * \return The minimum of this tensor and #rhs.
   * */
  Tensor min(const Tensor &rhs) const;
  Tensor min_(const Tensor &rhs) const;

  Tensor pow(double v) const { return pow(constant(v)); }
  Tensor pow_(double v) const { return pow_(constant(v)); }

  Tensor mul(double v) const { return mul(constant(v)); }
  Tensor mul_(double v) const { return mul_(constant(v)); }

  Tensor div(double v) const { return div(constant(v)); }
  Tensor div_(double v) const { return div_(constant(v)); }

  Tensor sub(double v) const { return sub(constant(v)); }
  Tensor sub_(double v) const { return sub_(constant(v)); }

  Tensor max(double v) const { return max(constant(v)); }
  Tensor max_(double v) const { return max_(constant(v)); }

  Tensor min(double v) const { return min(constant(v)); }
  Tensor min_(double v) const { return min_(constant(v)); }

  /**
   * \return The remainder when this tensor is divided by #rhs. #rhs must have
   *         the same dtype as this tensor.
   *
   * This elementwise operation is identical to fmod for floating point
   * numbers, as defined for C++.
   * */
  Tensor rem(const Tensor &rhs) const;
  Tensor rem_(const Tensor &rhs) const;

  Tensor modulo_(uint64_t v) const { return rem_(constant(v)); }
  Tensor modulo(uint64_t v) const { return rem(constant(v)); }

  Tensor tickModulo_(uint64_t m) const { return add_(1).modulo_(m); }

  /**
   * Copy the values in #rhs to this tensor. This tensor operation supports
   * numpy broadcasting, so #rhs must have a shape which is numpy
   * broadcastable to this tensor's shape.
   * */
  Tensor copyFrom_(const Tensor &rhs) const;
  Tensor update_(const Tensor &rhs) const { return copyFrom_(rhs); }

  /**
   * \return A boolean tensor which is true where this tensor is greater than
   *         #rhs. This tensor operation supports numpy broadcasting.
   * */
  Tensor greaterThan(const Tensor &rhs) const;

  /**
   * \return A boolean tensor which is true where this tensor is (bitwise)
   *         equal to #rhs. This tensor operation supports numpy broadcasting.
   * */
  Tensor equalTo(const Tensor &rhs) const;

  /**
   * Matrix multiply, using numpy broadcasting rules, see
   * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
   *
   * \param arg1 the right-hand side argument of the matrix multiply.
   * */
  Tensor
  matmul(const Tensor &rhs, DType outType, const MatMulOptions &) const;

  Tensor matmul(const Tensor &arg1) const {
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
  Tensor encodeOneHot01_(const Tensor &indices) const;

  /**
   * Similar to encodeOneHot01_ but instead of '0' the value of the scalar
   * tensor 'off' is used, and instead of '1' the value of the scalar tensor
   * 'on' is used. #off and #on must have the same numerical type as this
   * tensor.
   * */
  Tensor encodeOneHotOffOn_(const Tensor &indices,
                            const Tensor &off,
                            const Tensor &on) const;

  /**
   * The natural logarithm of this tensor.
   * */
  Tensor log_() const;
  Tensor log() const;

  /**
   * Negate all elements of this tensor.
   * */
  Tensor neg_() const;
  Tensor neg() const;

  /**
   * Cast this tensor to a tensor of type #outType.
   * */
  Tensor to(DType outType) const;

  /** Unary elementwise operations */

  /**
   * The absolute value of this tensor.
   * */
  Tensor abs_() const;
  Tensor abs() const;

  /**
   * The sine of this tensor.
   * */
  Tensor sin_() const;
  Tensor sin() const;

  /**
   * The cosine of this tensor.
   * */
  Tensor cos_() const;
  Tensor cos() const;

  /**
   * The sign, or signum, of this tensor:
   *
   *             = -1 if x < 0
   * signum(x)   =  0 if x = 0
   *             = +1 if x > 0.
   *
   * The returned tensor has the same dtype as this tensor.
   **/
  Tensor signum_() const;
  Tensor signum() const;

  /**
   * The square root of this tensor.
   * */
  Tensor sqrt_() const;
  Tensor sqrt() const;

  /**
   * e^(this tensor) where e is the transcendental value 2.718...
   * */
  Tensor exp_() const;
  Tensor exp() const;

  /**
   * The inverse of this tensor, 1/(this tensor).
   * */
  Tensor inv_() const;
  Tensor inv() const;

  /**
   * The rectified linear unit of this tensor.
   * */
  Tensor relu_() const { return mul_(greaterThan(constant(0)).to(dtype())); }
  Tensor relu() const { return mul(greaterThan(constant(0)).to(dtype())); }

  /**
   * Fill this tensor with the scalar value #vScalar.
   * */
  Tensor fill_(const HostTensor &vScalar) const;

  /**
   * Fill this tensor with 0.
   * */
  Tensor setToZero_() const { return fill_(HostTensor::scalar(dtype(), 0)); }
  Tensor zero_() const { return setToZero_(); }

  /**
   * Fill this tensor with the lowest possible value.
   * */
  Tensor setToLowest_() const;

  Tensor name(const std::string &n) const;

  static Tensors tensors(const TensorIds &ids, Graph &g);

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
  Tensor updateFromHost_(const Tensor &sourceOnHost,
                         const CopyBetweenHostAndIpuOptions &opts = {}) const;

  /**
   * Copy this host tensor to ipu. \sa updateFromHost_.
   * */
  Tensor hostToIpu(DeviceId ipuDestination,
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
  Tensor updateFromIpu_(const Tensor &sourceOnIpu,
                        const CopyBetweenHostAndIpuOptions &opts = {}) const;

  /**
   * Copy this ipu tensor to host. \sa updateFromIpu_.
   * */
  Tensor ipuToHost(CircularBufferCount,
                   const CopyBetweenHostAndIpuOptions & = {}) const;

  /**
   * \return The root reference tensor of this tensor. For tensors which are
   *         created with a call to #refTo_, this is the tensor on which
   *         #refTo_ was called. For all other tensors, this method returns
   *         the tensor itself. \sa RefFrom_ op.
   * */
  Tensor rootRef() const;

  /**
   * \sa Graph::refsExcludingSelf.
   * */
  Tensors refsExcludingSelf() const;

  /**
   * The ids of the tensors in #ts.
   * */
  static TensorIds tensorIds(const Tensors &ts);

  bool isRootRef() const { return id() == rootRef().id(); }

  uint64_t nDerivedRefs() const { return op().nDerivedRefs(outIndex()); }

  bool hasDerivedRefs() const { return nDerivedRefs() != 0; }

  void setName(const std::string &nm) const;

  Tensor inTensor(InIndex i) const;

  /**
   * \return A boolean tensor that is true where this tensor is strictly
   *         positive.
   * */
  Tensor isStrictlyPositive() const {
    return greaterThan(constant(dtype(), 0.0));
  }

  /**
   * This tensor is copied from the calling scope into a callee sub-graph,
   * defined by #ce. This method returns all of the tensors in the callee
   * sub-graph that it is copied to.
   * */
  Tensors dstsInCallee(const CallEvent &ce) const;

  /**
   * This tensor is the destination of a copy out of a callee subgraph. This
   * method returns the source of this copy.
   * */
  Tensor srcInCallee(uint64_t calleeIndex) const;

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
  Tensor dynamicMultiSlice(const Tensor &offset,
                           const Dimensions &dims,
                           const Shape &sizes) const;

  /**
   * This tensor is the slice tensor, and is updated inplace with sliceable.
   * It has the same shape as the output of dynamicMultiSlice.
   * */
  Tensor dynamicMultiSlice_(const Tensor &sliceable,
                            const Tensor &offsets,
                            const Dimensions &) const;

  /**
   * Similar to dynamicMultiSlice, but only 1 slice is taken. The output has
   * the same rank as this tensor. #offset is a a rank-1 tensor.
   * */
  Tensor dynamicSlice(const Tensor &offset,
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
  Tensor dynamicMultiUpdate_(const Tensor &slice,
                             const Tensor &offset,
                             const Dimensions &) const;

  /**
   * \sa dynamicMultiUpdate_
   *
   * \param slice A tensor which has the same rank as this tensor, and is
   *              smaller that this tensor in the dimensions #dims.
   *
   * \param offset A rank-1 tensor, of the same size as #dims.
   * */
  Tensor dynamicUpdate_(const Tensor &slice,
                        const Tensor &offset,
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

  Tensor updateAt_(const Tensor &slice, const Tensor &index) const {
    return dynamicMultiUpdate_(slice.reshape_(slice.shape().prependOnes(2)),
                               index.reshape_({1, 1}),
                               Dimensions{0});
  }

  /**
   * The inverse operation of pushToStash.
   * */
  Tensor popFromStash(const Tensor &index) { return dynamicAt(index); }

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
  Tensor dynamicMultiUpdateMax_(const Tensor &source,
                                const Tensor &offsets) const;

  /**
   * Ensure that the creator of this tensor is scheduled before the
   * creator of #after by inserting a topological constraint between the two.
   * */
  void before(const Tensor &after) const;

  /**
   * Append information about this tensor to #ostr.
   * */
  void append(std::ostream &ostr) const;

  std::string str() const;

  /**
   *
   * The softmax non-linearlity applied to this tensor, with the reduction in
   * dimension #redDim.
   *
   * The softmax of a tensor #t is defined as,
   *
   * def unstableSoftmax(t):
   *   return t.exp() / t.reduceSum(redDim)    (1)
   *
   * For numerical stability, the tensor #t can be conditioned,
   *
   * def stableSoftmax(t)
   *   ts  =  t - t.reduceMax(redDim)          (2)
   *   return unstableSoftmax(ts).
   *
   * \param ss Defines if stability conditioning (2) should be applied.
   * */
  enum class StableSoftmax { No = 0, Yes };
  Tensor softmax(uint64_t redDim,
                 StableSoftmax ss = StableSoftmax::Yes) const;

  /**
   * This tensor is rank-2 of shape (N, C). This tensor is not a probability
   * tensor - this method applies softmax to this tensor.
   *
   * \param labels A rank-1 tensor of shape (N,) with values in the range
   *               [0,C).
   *
   * \param ss Whether or not to use the stable version of softmax when
   *           computing the probabilities.
   * */
  NllGrad nllGrad(const Tensor &labels,
                  StableSoftmax ss = StableSoftmax::Yes) const;

  /**
   * Helper methods for generating random host tensors.
   * */
  HostTensor uniformFloat64(double l, double u, uint32_t seed) const {
    return HostTensor::uniformFloat64(l, u, shape(), seed);
  }
  HostTensor uniformFloat32(double l, double u, uint32_t seed) const {
    return HostTensor::uniformFloat32(l, u, shape(), seed);
  }
  HostTensor randomInt32(int32_t low, int32_t upp, uint32_t seed) {
    return HostTensor::randomInt32(low, upp, shape(), seed);
  }
  /**
   * Copy slices of this remote tensor to ipu.
   *
   * This method perfoms a copy of data from the remote device on which this
   * tensor is located, to the ipu on which the #indices tensor is located.
   *
   * Remote tensors must be rank-2, with dimension-0 being #repeats and
   * dimension-1 being #numElements. The returned tensor is also rank-2, of
   * shape (indices.size(), #numElements).
   *
   * \param indices a rank-1 integral ipu tensor.
   */
  Tensor remoteToIpu(const Tensor &indices) const;

  /**
   * This method can be called on remote tensors with dim(0)=1. It copies the
   * tensor to the remote buffer's corresponding ipu. The destination has the
   * same shape as this tensor.
   * */
  Tensor remoteToIpu() const;

  /**
   * Update the values in this ipu tensor with values from the remote tensor
   * #remoteTensor. Where #remoteTensor has shape (n0, S), this tensor has
   * shape (n1, S), and #indices has shapee (n1), #indices maps rows in
   * #remoteTensor to destination rows in this ipu tensor.
   * */
  Tensor updateIpuFromRemote_(const Tensor &remoteTensor,
                              const Tensor &indices) const;

  /**
   * The inverse of #updateIpuFromRemote_. All the shapes are identically
   * defined, the only difference is that the copy is from ipu to remote.
   * */
  Tensor updateRemoteFromIpu_(const Tensor &ipuTensor,
                              const Tensor &indices) const;

  /**
   * Copy rows from this rank-2 ipu tensor to a remote tensor. The remote
   * tensor created will have shape (nRepeats, dim(1)). The tensor #indices
   * has shape (dim(0)) and it defines the rows of this ipu tensor to copy to
   * the remote device.
   * */
  Tensor ipuToRemote(const Tensor &indices,
                     uint64_t nRepeats,
                     const RemoteOptions &) const;

  /**
   * The inverse of remoteToIpu.
   * */
  Tensor ipuToRemote(const RemoteOptions &) const;

private:
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
  Tensor createTensor(const TensorIds &ins,
                      const TensorInfos &outs,
                      Args &&...args) const {
    return {{createComputeOp<TOp>(ins, outs, std::forward<Args>(args)...), 0},
            &graph()};
  }

  template <class TOp, class... Args>
  Tensor createUnaryWithSameInfo(Args &&...args) const {
    return createTensor<TOp>({id()}, {info()}, std::forward<Args>(args)...);
  }

  template <class TOp, class... Args>
  Tensor createWithNumpyShape(const TensorIds &ins, Args &&...args) const;

  template <class TOp, class... Args>
  Tensor createBooleanWithNumpyShape(const TensorIds &ins,
                                     Args &&...args) const;

  /**
   * Create a tensor by applying a one-to-one view-change op of type TOp to
   * this tensor. The created tensor has shape #outShape. If the view-change
   * is effectively the identity view-change, then no new op is created in the
   * graph, and this tensor is returned directly (so that the returned tensor
   * has same id as this tensor).
   * */
  template <class TOp, class... Args>
  Tensor createUnaryViewChange(const Shape &outShape, Args &&...args) const {

    // If the view-change is an identity view, do not create a new tensor --
    // just return this tensor.
    if (TOp::isIdentity(shape(), outShape, args...)) {
      return Tensor(id(), &graph());
    }
    return createTensor<TOp>(
        {id()}, {info().withShape(outShape)}, std::forward<Args>(args)...);
  }

  template <class TOp, class... Args>
  Tensor createUnaryWithNewShape(const Shape &s, Args &&...args) const {
    return createTensor<TOp>(
        {id()}, {info().withShape(s)}, std::forward<Args>(args)...);
  }

  TensorId id_;
  Graph *pGraph_;

  template <typename X> [[noreturn]] static void err(X &&x) {
    throw poprithms::error::error("common::compute", x);
  }

  const Op &op(OpId opId) const;
};

inline Tensor operator*(const Tensor &a, const Tensor &b) { return a.mul(b); }

inline Tensor operator+(const Tensor &a, const Tensor &b) { return a.add(b); }

inline Tensor operator/(const Tensor &a, const Tensor &b) { return a.div(b); }

inline Tensor operator-(const Tensor &a, const Tensor &b) { return a.sub(b); }

inline Tensor concat_(const Tensors &ts, uint64_t axis) {
  return Tensor::concat_(ts, axis);
}

inline Tensor matmul(const Tensor &t0, const Tensor &t1) {
  return t0.matmul(t1);
}

inline std::ostream &operator<<(std::ostream &ost, const OptionalTensor &ot) {
  ot.append(ost);
  return ost;
}

inline std::ostream &operator<<(std::ostream &ost,
                                const OptionalTensors &ots) {
  poprithms::util::append(ost, ots);
  return ost;
}

inline std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  t.append(os);
  return os;
}

/**
 * The result of applying (1) softmax and then (2) negative log-likelihood
 * to a tensor #In.
 * */
struct NllGrad {
public:
  /**
   * Negative log-likelihood loss.
   * */
  Tensor loss() const { return loss_; }

  /**
   * dLoss/dIn, where
   *   Loss = nll(probs=In.softmax(), labels)
   **/
  Tensor dIn() const { return dIn_; }

private:
  Tensor loss_;
  Tensor dIn_;
  friend class Tensor;
  NllGrad(Tensor loss, Tensor dIn) : loss_(loss), dIn_(dIn) {}
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
