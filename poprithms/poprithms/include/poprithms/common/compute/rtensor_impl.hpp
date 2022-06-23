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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
