// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/autodiff/core/summary.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

TensorIds Summary::allTensorIds() const {
  TensorIds x = gradsIn_;
  x.insert(x.end(), checkpointsIn_.cbegin(), checkpointsIn_.cend());
  x.insert(x.end(), targetGrads_.cbegin(), targetGrads_.cend());
  return x;
}

} // namespace core

} // namespace autodiff
} // namespace poprithms
