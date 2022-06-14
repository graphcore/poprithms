// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TENSOR_HPP
#define POPRITHMS_COMMON_COMPUTE_TENSOR_HPP

#include <poprithms/common/compute/rtensor.hpp>
#include <poprithms/common/multiout/toptionaltensor.hpp>

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

using Tensors = std::vector<Tensor>;

using OptionalTensor  = poprithms::common::multiout::TOptionalTensor<Tensor>;
using OptionalTensors = std::vector<OptionalTensor>;

} // namespace compute

namespace multiout {
extern template class TOptionalTensor<compute::Tensor>;
}
} // namespace common
} // namespace poprithms

#endif