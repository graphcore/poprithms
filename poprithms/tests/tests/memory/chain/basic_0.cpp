// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>

namespace {

using namespace poprithms::memory::chain;

} // namespace

int main() {
  Chain chain({5, 6});
  chain.reverse(Dimensions({0, 1}))
      .dimShuffle({{1, 0}})
      .reshape({30, 1})
      .expand({30, 3})
      .slice({5, 0}, {25, 3})
      .settFillInto(Stride(3), Dimension(0));
  std::cout << "logging" << std::endl;
  std::cout << chain << std::endl;
  return 0;
}
