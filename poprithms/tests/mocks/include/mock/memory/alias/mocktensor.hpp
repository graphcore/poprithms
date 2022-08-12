// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_HPP
#define MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_HPP

#include <gmock/gmock.h>

#include <poprithms/memory/alias/tensor.hpp>

namespace mock::poprithms::memory::alias {

class MockTensor {
public:
  MockTensor();
  virtual ~MockTensor();

  MOCK_METHOD(const ::poprithms::ndarray::Shape &, shape, (), (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              reshape,
              (const ::poprithms::ndarray::Shape &),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              dimShuffle,
              (const ::poprithms::util::Permutation &),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              slice,
              (uint64_t, uint64_t, ::poprithms::ndarray::Dimension),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              slice,
              (const ::poprithms::memory::alias::Lower &,
               const ::poprithms::memory::alias::Upper &),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              subscript,
              (uint64_t),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              expand,
              (const ::poprithms::ndarray::Shape &),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor, flatten, (), (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              reverse,
              (uint64_t),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              reverse,
              (const std::vector<uint64_t> &),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor, squeeze, (), (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              broadcast,
              (int64_t, uint64_t),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              subsample,
              (int64_t, uint64_t),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              upsample,
              (uint64_t, uint64_t),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              index,
              (const std::vector<uint64_t> &),
              (const));
};

extern MockTensor *mockAliasTensor_;

} // namespace mock::poprithms::memory::alias
#endif // MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_HPP
