// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_UTIL_PERMUTATION_HPP
#define MOCKS_POPRITHMS_UTIL_PERMUTATION_HPP

#include <gmock/gmock.h>

#include <poprithms/util/permutation.hpp>

namespace mock::poprithms::util {

class MockPermutation {
public:
  MockPermutation();
  virtual ~MockPermutation();

  MOCK_METHOD(::poprithms::util::Permutation,
              dimShufflePartial,
              (uint64_t,
               const std::vector<uint64_t> &,
               const std::vector<uint64_t> &));
};

extern MockPermutation *mockAliasPermutation_;

} // namespace mock::poprithms::util
#endif // MOCKS_POPRITHMS_UTIL_PERMUTATION_HPP
