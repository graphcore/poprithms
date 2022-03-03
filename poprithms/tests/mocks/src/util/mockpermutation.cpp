// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mock/util/mockpermutation.hpp>

#include <poprithms/util/permutation.hpp>

namespace mock::poprithms::util {
MockPermutation *mockAliasPermutation_ = nullptr;
}

namespace poprithms::util {
poprithms::util::Permutation
Permutation::dimShufflePartial(uint64_t rnk,
                               const std::vector<uint64_t> &src,
                               const std::vector<uint64_t> &dst) {
  return mock::poprithms::util::mockAliasPermutation_->dimShufflePartial(
      rnk, src, dst);
}
} // namespace poprithms::util
