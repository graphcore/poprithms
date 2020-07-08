// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/util/error.hpp>
#include <poprithms/util/permutation.hpp>

int main() {
  using namespace poprithms::util;
  Permutation p({1, 2, 0, 4, 5, 3});
  const auto inv = p.inverse();
  if (inv != Permutation({2, 0, 1, 5, 3, 4})) {
    throw error("Unexpected inverse in Permutation test");
  }
  if (inv.isIdentity()) {
    throw error("This Permutation is not identity, test failure");
  }
  const auto permuted = p.apply(std::vector<int>{13, 11, 7, 5, 3, 2});
  if (permuted != std::vector<int>{11, 7, 13, 3, 2, 5}) {
    throw error("Permuted vector is not as expected");
  }
  return 0;
}
