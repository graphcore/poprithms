// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RSUBGRAPH_IMPL
#define POPRITHMS_COMMON_COMPUTE_RSUBGRAPH_IMPL

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/common/compute/rsubgraph.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>
#include <poprithms/program/callstack/copyin.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::ndarray::TensorInfo;
using poprithms::ndarray::TensorInfos;
using poprithms::program::callstack::CopyIns;

template <typename T>
T RSubGraph<T>::constant(const HostTensor &t, DeviceId deviceId) {
  TensorInfo outInfo{t.shape(), deviceId, t.dtype()};
  const auto opId = graph().nxtOpId();
  graph().template createComputeOp<ConstInit>({}, id(), {outInfo}, t);
  return {TensorId(opId, OutIndex(0)), &graph()};
}

template <typename T>
T RSubGraph<T>::variable(DType t, const Shape &s, DeviceId d) {
  auto opId =
      graph().template createComputeOp<poprithms::common::compute::VarInit>(
          {}, id(), {{s, d, t}});
  return {{opId, 0}, &graph()};
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
