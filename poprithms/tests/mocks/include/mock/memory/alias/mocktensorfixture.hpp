// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_FIXTURE_HPP
#define MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_FIXTURE_HPP

#include <mock/memory/alias/mocktensor.hpp>

namespace mock::poprithms::memory::alias {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockTensorFixture {
public:
  MockTensorFixture() {
    mock::poprithms::memory::alias::mockAliasTensor_ =
        static_cast<mock::poprithms::memory::alias::MockTensor *>(
            &mockAliasTensor);
  }

  ~MockTensorFixture() {
    mock::poprithms::memory::alias::mockAliasTensor_ = nullptr;
  }

protected:
  Mock<MockTensor> mockAliasTensor;
};

} // namespace mock::poprithms::memory::alias

#endif // MOCKS_POPRITHMS_MEMORY_ALIAS_TENSOR_FIXTURE_HPP
