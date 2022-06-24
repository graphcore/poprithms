// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP
#define POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP

#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/binaryelementwise.hpp>
#include <poprithms/common/compute/ops/reduce.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/rtensor.hpp>
#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace compute {

template <typename T>
T RTensor<T>::constant(const poprithms::compute::host::Tensor &t) const {
  return subGraph().constant(t, deviceId());
}

template <typename T>
T RTensor<T>::constant(SubGraphId sgId, double v) const {
  return RSubGraph<T>(sgId, graph()).constant(dtype(), v, deviceId());
}

template <typename T> RSubGraph<T> RTensor<T>::subGraph() const {
  return RSubGraph<T>(subGraphId(), graph());
}

template <typename T> T RTensor<T>::constant(DType d, double v) const {
  return constant(poprithms::compute::host::scalar(d, v));
}

template <typename T> T RTensor<T>::div_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Div_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::div(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Div>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::pow_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Pow_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::pow(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Pow>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::copyFrom_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<CopyFrom_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::greaterThan(const RTensor<T> &rhs) const {

  TensorInfo oi(Shape::numpyVariadic({shape(), rhs.shape()}),
                deviceId(),
                DType::Boolean);
  return createTensor<GreaterThan>({id(), rhs.id()}, oi);
}

template <typename T> T RTensor<T>::sub_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Sub_>({id(), rhs.id()});
}
template <typename T> T RTensor<T>::sub(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Sub>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::log_() const {
  return createUnaryWithSameInfo<Log_>();
}
template <typename T> T RTensor<T>::log() const {
  return createUnaryWithSameInfo<Log>();
}

template <typename T> T RTensor<T>::dstInCaller(const CallEvent &ce) const {
  TensorId dst = graph().dstInCaller(id_, ce);
  return {dst, &graph()};
}

template <typename T> T RTensor<T>::srcInCaller(const CallEvent &cse) const {
  return {graph().srcInCaller(id_, cse), &graph()};
}

template <typename T> T RTensor<T>::dstInCaller(OpId call) const {
  return dstInCaller(graph().callEvent(call));
}

template <typename T>
RTensor<T>::RTensor(const TensorId &tId, Graph *m) : id_(tId), pGraph_(m) {}

template <typename T> T RTensor<T>::refTo_(SubGraphId destination) const {
  auto x = graph().template tRefFrom<RefFrom>(id(), destination);
  return {x, &graph()};
}

template <class T>
template <class TOp, class... Args>
OpId RTensor<T>::createComputeOp(const TensorIds &inIds_,
                                 const TensorInfos &outInfos,
                                 Args &&...args) const {

  if (inIds_.empty()) {
    std::ostringstream oss;
    oss << "Failed to use Tensor::createComputeOp without any inputs. "
        << "Inputs are required to determine the sub-graph of the output. "
        << "This case must be handled using Graph::createComputeOp "
        << "directly. ";
    throw poprithms::error::error("common::compute", oss.str());
  }
  auto sgId = graph().subGraphId(inIds_.at(0));
  return graph().template createComputeOp<TOp>(
      inIds_, sgId, outInfos, std::forward<Args>(args)...);
}

template <typename T> T RTensor<T>::reshape_(const Shape &s) const {
  return createUnaryViewChange<Reshape_>(s);
}

template <typename T> T RTensor<T>::expand_(const Shape &s) const {
  return createUnaryViewChange<Expand_>(s);
}

template <typename T> TensorInfo RTensor<T>::info() const {
  return graph().tensorInfo(id());
}

template <typename T> T RTensor<T>::dimShuffle_(const Permutation &p) const {
  return createUnaryViewChange<DimShuffle_>(p.apply(shape().get()), p);
}

template <typename T> T RTensor<T>::reverse_(const Dimensions &dims_) const {

  // Canonicalize the reverse dimensions (ascending order, repetitions
  // reduced modulo 2).
  const auto revDimsCanonical =
      shape().getCanonicalReverseIndices(dims_.get());

  return createUnaryViewChange<Reverse_>(
      shape().get(), Dimensions(std::move(revDimsCanonical)));
}

template <typename T> T RTensor<T>::reduce(const CommutativeOp cop) const {
  return reduce(Shape({}), cop);
}

template <typename T>
T RTensor<T>::reduce(const Shape &outShape, const CommutativeOp cop) const {
  auto out_ = reduce(shape().reductionDimensions(outShape), cop);
  return out_.reshape_(outShape);
}

template <typename T>
T RTensor<T>::reduce(Dimension d, const CommutativeOp cop) const {
  Dimensions dims({d});
  return reduce(dims, cop);
}

template <typename T>
T RTensor<T>::reduce(const Dimensions &d, CommutativeOp cop) const {

  if (d.empty()) {
    // we cannot just return this tensor, because there must at least be a
    // copy here.
  }
  auto outShape = shape().get();
  for (uint64_t i = 0; i < d.get().size(); ++i) {
    if (d.at(i).get() >= shape().rank_u64()) {
      throw poprithms::error::error("common::compute",
                                    "Invalid dimension in reduceSum");
    }
    outShape[d.at(i).get()] = 1;
  }

  switch (cop) {
  case (CommutativeOp::Sum): {
    return createUnaryWithNewShape<ReduceSum>(outShape, d);
  }

  case (CommutativeOp::Min): {
    return createUnaryWithNewShape<ReduceMin>(outShape, d);
  }

  case (CommutativeOp::Max): {
    return createUnaryWithNewShape<ReduceMax>(outShape, d);
  }

  case (CommutativeOp::Product): {
    return createUnaryWithNewShape<ReduceProduct>(outShape, d);
  }
  default:
    throw poprithms::error::error("common::compute",
                                  "Unrecognised reduction type");
  }
}

template <typename T> T RTensor<T>::mul_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Mul_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::mul(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Mul>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::add_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Add_>({id(), rhs.id()});
}
template <typename T> T RTensor<T>::add(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Add>({id(), rhs.id()});
}

template <typename T>
T RTensor<T>::padWithBroadcastConstZero_(const Lower &l,
                                         const Upper &u) const {

  auto z = constant(0.);

  // progressively pad the tensor, one dimension at a time. The padding on all
  // edges is all an alias of the constant 'z' above.

  auto t = T(id(), &graph());
  for (uint64_t d = 0; d < rank_u64(); ++d) {

    // concatate the tensors (lower_padding, t, upper_padding). If the padding
    // is zero, then leave it off.
    std::vector<T> toConcat;
    if (l.at(d) > 0) {
      toConcat.push_back(z.expand_(t.shape().resizeSingleDim(l.at(d), d)));
    }

    toConcat.push_back(t);

    if (u.at(d) > 0) {
      toConcat.push_back(z.expand_(t.shape().resizeSingleDim(u.at(d), d)));
    }

    t = concat_(toConcat, d);
  }

  return t;
}

template <typename T>
T RTensor<T>::slice_(const Lower &l, const Upper &u) const {

  const auto outShape = shape().slice(l, u);

  // Check for a slice which doesn't slice anything out. The second condition
  // here is to confirm that the bounds are valid (lower is 0s).
  if (outShape == shape() && Shape(u) == shape()) {
    return {id(), &graph()};
  }

  return createUnaryWithNewShape<Slice_>(outShape, l, u);
}

template <typename T>
T RTensor<T>::slice_(const Dimensions &dims,
                     const std::vector<uint64_t> &starts_,
                     const std::vector<uint64_t> &ends_) const {
  auto lu = shape().getFullSliceBounds(dims, starts_, ends_);
  return slice_(lu.first, lu.second);
}

template <typename T>
T RTensor<T>::slice(const Dimensions &a,
                    const std::vector<uint64_t> &b,
                    const std::vector<uint64_t> &c) const {
  return slice_(a, b, c).copy();
}

template <typename T> T RTensor<T>::copy(DeviceId target) const {

  const auto targetType = graph().device(target).deviceType();
  if (targetType != deviceType()) {
    std::ostringstream oss;
    oss << "Tensor::copy(target=" << target
        << ") is invalid, as this tensor, " << id()
        << " has a different device type. "
        << "This method can only copy Ipu->Ipu. " << deviceType() << "->"
        << targetType;
    throw poprithms::error::error("common::compute", oss.str());
  }

  auto targetTensor = variable(target);
  return targetTensor.copyFrom_(*this);
}

template <typename T>
T RTensor<T>::slice(const Lower &l, const Upper &u) const {
  return slice_(l, u).copy();
}

template <typename T>
T RTensor<T>::slice(Dimension d, int64_t l, int64_t u) const {
  return slice_(d, l, u).copy();
}

template <typename T>
T RTensor<T>::slice_(Dimension d, int64_t l, int64_t u) const {
  if (l < 0 || u < 0) {
    std::ostringstream oss;
    oss << "Invalid call, Tensor::slice_ (Dimension = " << d.get()
        << ", l = " << l << ", u = " << u << "). "
        << "The lower (l) and upper (u) bounds must both be non-negative. ";
    throw poprithms::error::error("common::compute", oss.str());
  }

  const auto fullSliceBounds = shape().getFullSliceBounds(
      d, static_cast<uint64_t>(l), static_cast<uint64_t>(u));

  return slice_(std::get<0>(fullSliceBounds), std::get<1>(fullSliceBounds));
}

template <typename T> T RTensor<T>::reverse(const Dimensions &ds) const {
  return reverse_(ds).copy();
}

template <typename T> T RTensor<T>::reverse(uint64_t d) const {
  return reverse_(d).copy();
}

template <typename T> T RTensor<T>::dimShuffleFinalTwo() const {
  return dimShuffle(Permutation::reverseFinalTwo(rank_u64()));
}

template <typename T>
T RTensor<T>::dimRoll(uint64_t from, uint64_t to) const {
  return dimShuffle(Permutation::dimRoll(rank_u64(), {from, to}));
}

template <typename T>
T RTensor<T>::dimRoll_(uint64_t from, uint64_t to) const {
  return dimShuffle_(Permutation::dimRoll(rank_u64(), {from, to}));
}

template <typename T> T RTensor<T>::dimShuffleFinalTwo_() const {
  return dimShuffle_(Permutation::reverseFinalTwo(rank_u64()));
}

template <typename T>
T RTensor<T>::squeeze(const std::vector<uint64_t> &dims) const {
  return reshape(shape().squeeze(dims));
}

template <typename T>
T RTensor<T>::squeeze_(const std::vector<uint64_t> &dims) const {
  return reshape_(shape().squeeze(dims));
}

template <typename T> T RTensor<T>::variable(const Shape &s0) const {
  return subGraph().variable(dtype(), s0, deviceId());
}

template <typename T>
T RTensor<T>::variable(const Shape &s0, DeviceId dId) const {
  return subGraph().variable(dtype(), s0, dId);
}

template <typename T> T RTensor<T>::variable(DType t, const Shape &s) const {
  return subGraph().variable(t, s, deviceId());
}

template <typename T> T RTensor<T>::variable(SubGraphId sgId) const {
  return RSubGraph<T>(sgId, graph()).variable(dtype(), shape(), deviceId());
}

template <typename T> T RTensor<T>::variable() const {
  return subGraph().variable(dtype(), shape(), deviceId());
}

template <typename T> T RTensor<T>::variable(DeviceId did) const {
  return subGraph().variable(dtype(), shape(), did);
}

template <typename T>
T RTensor<T>::concat_(const std::vector<T> &ts, uint64_t axis) {

  if (ts.size() == 0) {
    throw poprithms::error::error(
        "common::compute", "cannot concatenate empty vector of Tensors");
  }

  // If there is just 1 tensor being concatenated, return it.
  if (ts.size() == 1) {
    return T(ts[0].id(), &ts[0].graph());
  }

  auto &m  = ts[0].graph();
  auto ids = TSlickConverter::getIds(ts);
  auto out = ts[0].template createTensor<Concat_>(
      ids,
      {m.tensorInfo(ts[0]).withShape(Shape::concat(m.shapes(ids), axis))},
      axis);

  return out;
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
