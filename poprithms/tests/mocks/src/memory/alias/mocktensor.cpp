// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mock/memory/alias/mocktensor.hpp>

#include <poprithms/memory/alias/tensor.hpp>

namespace mock::poprithms::memory::alias {
MockTensor *mockAliasTensor_ = nullptr;
} // namespace mock::poprithms::memory::alias

namespace poprithms::memory::alias {
const Shape &Tensor::shape() const {
  return mock::poprithms::memory::alias::mockAliasTensor_->shape();
}

Tensor Tensor::reshape(const Shape &shape) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->reshape(shape);
}

Tensor Tensor::dimShuffle(const Permutation &perm) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->dimShuffle(perm);
}

Tensor Tensor::subscript(uint64_t index) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->subscript(index);
}

Tensor Tensor::slice(uint64_t start, uint64_t end, Dimension sliceDim) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->slice(
      start, end, sliceDim);
}

Tensor Tensor::slice(const Lower &l, const Upper &u) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->slice(l, u);
}

Tensor Tensor::expand(const Shape &to) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->expand(to);
}
Tensor Tensor::flatten() const {
  return mock::poprithms::memory::alias::mockAliasTensor_->flatten();
}

Tensor Tensor::reverse(uint64_t dimension) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->reverse(dimension);
}

Tensor Tensor::reverse(const std::vector<uint64_t> &dimensions) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->reverse(
      dimensions);
}

Tensor Tensor::squeeze() const {
  return mock::poprithms::memory::alias::mockAliasTensor_->squeeze();
}

Tensor Tensor::broadcast(int64_t N, uint64_t dimension) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->broadcast(
      N, dimension);
}

Tensor Tensor::subsample(int64_t stride, uint64_t dimension) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->subsample(
      stride, dimension);
}

Tensor Tensor::upsample(uint64_t scale, uint64_t dim) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->upsample(scale,
                                                                    dim);
}

Tensor Tensor::index(const std::vector<uint64_t> &indices) const {
  return mock::poprithms::memory::alias::mockAliasTensor_->index(indices);
}

bool Tensor::containsColor(Color c) const {
  return c.get() == 1 || c.get() == 0;
}
} // namespace poprithms::memory::alias
