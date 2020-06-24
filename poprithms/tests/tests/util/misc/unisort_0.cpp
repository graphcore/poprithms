// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/util/error.hpp>
#include <poprithms/util/unisort.hpp>

int main() {
  using namespace poprithms::util;
  std::vector<char> foo{'g', 'a', 'g', 'f', 'o', 'o', 'o'};
  auto bar = unisorted(foo);
  if (bar != std::vector<char>{'a', 'f', 'g', 'o'}) {
    throw error("unisorted test failed to unique-ify and sort char vector");
  }
  return 0;
}
