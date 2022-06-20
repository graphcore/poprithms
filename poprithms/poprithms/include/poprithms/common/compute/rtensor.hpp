// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RTENSOR_HPP
#define POPRITHMS_COMMON_COMPUTE_RTENSOR_HPP

#include <memory>
#include <vector>

#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/common/compute/replication.hpp>
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
using common::compute::ReplicationFactor;
using common::multiout::OpId;
using common::multiout::OpIds;
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
using poprithms::ndarray::Offset;
using poprithms::ndarray::Offsets;
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

  /**
   * \return The shape of this tensor.
   * */
  Shape shape() const;

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
   * \return An alias of this tensor. The returned tensor has the same rank as
   *         this tensor, but with the dimensions of this tensor permuted by
   *         #permutation.
   * */
  T dimShuffle_(const Permutation &permutation) const;

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
   * Broadcast this tensor along the dimensions necessary to create a tensor
   * of shape #expandedShape.
   *
   * \param expandedShape The shape of the expanded view of this tensor.
   *                      #expandedShape must numpy-dominate the shape of this
   *                      tensor. For example this tensor might have shape
   *                      (4,1) and #expandedShape might be (3,4,5).
   * */
  T expand_(const Shape &expandedShape) const;

  /**
   * \return A scalar Tensor, whose value is the reduction of all elements
   *         in the Tensor using the commutative op #cop.
   */

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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
