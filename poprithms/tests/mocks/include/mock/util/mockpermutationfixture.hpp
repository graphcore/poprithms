// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_UTIL_PERMUTATION_FIXTURE_HPP
#define MOCKS_POPRITHMS_UTIL_PERMUTATION_FIXTURE_HPP

#include <mock/util/mockpermutation.hpp>

namespace mock::poprithms::util {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockPermutationFixture {
public:
  MockPermutationFixture() {
    mock::poprithms::util::mockAliasPermutation_ =
        static_cast<mock::poprithms::util::MockPermutation *>(
            &mockPermutation);
  }

  ~MockPermutationFixture() {
    mock::poprithms::util::mockAliasPermutation_ = nullptr;
  }

protected:
  Mock<MockPermutation> mockPermutation;
};

} // namespace mock::poprithms::util

#endif // MOCKS_POPRITHMS_UTIL_PERMUTATION_FIXTURE_HPP
