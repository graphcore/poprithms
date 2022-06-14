// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP
#define POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP

#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/rtensor.hpp>
#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace compute {

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

template <typename T> Shape RTensor<T>::shape() const {
  return graph().shape(id());
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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
