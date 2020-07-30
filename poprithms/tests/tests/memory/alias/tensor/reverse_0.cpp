// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {

  Graph g;

  const auto alloc1 = g.tensor(g.allocate({10, 20, 30}));
  const auto r0     = alloc1.reverse(0);
  const auto r1     = alloc1.reverse(1);
  const auto sl0    = alloc1.slice({0, 0, 0}, {5, 20, 30});
  const auto sl1    = r0.slice({0, 0, 0}, {5, 20, 30});
  const auto sl2    = r0.slice({3, 0, 0}, {6, 20, 30});
  if (sl0.intersectsWith(sl1)) {
    throw error("No intersection with half-mirror expected");
  }

  if (!sl0.intersectsWith(sl2)) {
    throw error("Intersection with flipped half-mirror expected");
  }
  if (!r1.intersectsWith(sl1) || !r1.intersectsWith(sl2)) {
    throw error(
        "Intersection with sliced flips along other dimensions expected");
  }

  const auto r2      = alloc1.reverse({0, 1, 2});
  const auto sample0 = r2.slice({1, 1, 1}, {10, 20, 30})
                           .subsample(2, 0)
                           .subsample(2, 1)
                           .subsample(2, 2);
  const auto sample1 = alloc1.subsample(2, 0).subsample(2, 1).subsample(2, 2);
  if (!sample0.intersectsWith(sample1)) {
    throw error("Expected intersection");
  }
  std::cout << g << std::endl;
}

} // namespace

int main() {
  test0();
  return 0;
}
