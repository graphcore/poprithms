// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mock/memory/alias/mocktensor.hpp>

#include <poprithms/memory/alias/tensor.hpp>

namespace mock::poprithms::memory::alias {
MockTensor *mockAliasTensor_ = nullptr;
} // namespace mock::poprithms::memory::alias

namespace poprithms::memory::alias {
Tensor Tensor::reshape(const Shape &shape) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->reshape(shape);
}

Tensor Tensor::dimShuffle(const Permutation &perm) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->dimShuffle(perm);
}
} // namespace poprithms::memory::alias
