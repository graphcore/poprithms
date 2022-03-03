// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_HPP
#define MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_HPP

#include <gmock/gmock.h>

#include <poprithms/memory/alias/tensor.hpp>

namespace mock::poprithms::memory::alias {

class MockTensor {
public:
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              reshape,
              (::poprithms::ndarray::Shape),
              (const));
  MOCK_METHOD(::poprithms::memory::alias::Tensor,
              dimShuffle,
              (const ::poprithms::util::Permutation &),
              (const));
};

extern MockTensor *mockAliasTensor_;

} // namespace mock::poprithms::memory::alias
#endif // MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_HPP
