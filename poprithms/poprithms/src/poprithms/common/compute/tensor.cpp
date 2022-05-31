// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/rtensor.hpp>
#include <poprithms/common/compute/rtensor_impl.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

template class RTensor<Tensor>;

Tensor::Tensor(const TensorId &tId, Graph *pGraph) : RTensor(tId, pGraph) {}

} // namespace compute
} // namespace common
} // namespace poprithms
