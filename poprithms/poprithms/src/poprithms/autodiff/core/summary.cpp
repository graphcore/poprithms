// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/autodiff/core/summary.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

void Summary::setGradsIn(const TensorIds &ids) { gradsIn_ = ids; }
void Summary::setCheckpointsIn(const TensorIds &ids) { checkpointsIn_ = ids; }
void Summary::setTargetGrads(const TensorIds &ids) { targetGrads_ = ids; }

} // namespace core

} // namespace autodiff
} // namespace poprithms
