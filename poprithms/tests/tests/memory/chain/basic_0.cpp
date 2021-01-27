// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>

namespace {

using namespace poprithms::memory::chain;

} // namespace

int main() {
  Chain chain({5, 6});
  chain.reverse(Dimensions({0, 1}));
  chain.dimShuffle({{1, 0}});
  chain.reshape({30, 1});
  chain.expand({30, 3});
  chain.slice({5, 0}, {25, 3});
  chain.settFillInto(Stride(3), Dimension(0));
  return 0;
}
