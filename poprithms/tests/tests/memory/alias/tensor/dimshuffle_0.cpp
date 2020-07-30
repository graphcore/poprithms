// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

int main() {

  using namespace poprithms::memory::alias;
  using namespace poprithms::memory::nest;
  Graph g;
  const auto shf = g.tensor(g.allocate({4, 3, 2})).dimshuffle({{1, 2, 0}});
  if (shf.shape() != Shape{3, 2, 4}) {
    throw error("Failed in dimshuffle basic test");
  }
}
