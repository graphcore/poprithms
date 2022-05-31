// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP
#define POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/rtensor.hpp>

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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
