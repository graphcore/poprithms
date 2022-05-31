// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TENSOR_HPP
#define POPRITHMS_COMMON_COMPUTE_TENSOR_HPP

#include <poprithms/common/compute/rtensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Graph;
class Tensor;
class SubGraph;
extern template class RTensor<Tensor>;

template <typename T> class RSubGraph;

/**
 * See the RTensor template class for information about this class.
 * */
class Tensor : public RTensor<Tensor> {

public:
  Tensor() = delete;
  Tensor(const TensorId &, Graph *);
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
