// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/util/unisort.hpp>

int main() {
  using namespace poprithms;
  using namespace poprithms::schedule::anneal;
  std::vector<char> foo{'g', 'a', 'g', 'f', 'o', 'o', 'o'};
  auto bar = util::unisorted(foo);
  if (bar != std::vector<char>{'a', 'f', 'g', 'o'}) {
    throw error("unisorted test failed to unique-ify and sort char vector");
  }
  return 0;
}
