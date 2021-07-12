// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <map>
#include <set>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
namespace {
using namespace poprithms::memory::alias;
void testToConcat() {
  Graph g;
  const auto id0 = g.allocate({3, 4}, 0);
  const auto id1 = g.allocate({6, 4}, 3);
  g.allocationToConcat({id0, id0}, 0, id1);
  if (!g.containsAliases(id1)) {
    throw poprithms::test::error(
        "After the transform to concat, id1 does containt aliases");
  }
  if (g.containsColor(id1, 3)) {
    throw poprithms::test::error("id1 is now of color 0");
  }
}

void testToSettSample() {
  Graph g;
  const auto id0 = g.allocate({10, 10}, 0);
  const auto id1 = g.allocate({5, 5}, 1);
  const auto id2 = g.tensor(id0)
                       .settSample({{10, 10}, {{{{1, 1, 1}}}, {{{1, 1, 0}}}}})
                       .id();

  g.allocationToSettsample(
      id0, {{10, 10}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}}}, id1);

  if (g.areAliased(id1, id2) || !g.areAliased(id0, id1) ||
      !g.areAliased(id0, id2)) {
    throw poprithms::test::error("error in testToSettSample");
  }
}

void testToDimShuffle() {
  Graph g;
  const auto id0 = g.allocate({2, 3, 4}, 0);
  const auto id1 = g.allocate({3, 4, 2}, 0);
  g.allocationToDimshuffle(id0, {{1, 2, 0}}, id1);
  if (!g.areAliased(id0, id1)) {
    throw poprithms::test::error("error in testToDimShuffle");
  }
}

void testToReshape() {
  Graph g;
  const auto id0 = g.allocate({2, 3}, 0);
  const auto id1 = g.allocate({6}, 0);
  g.allocationToReshape(id0, id1);
  if (!g.areAliased(id0, id1)) {
    throw poprithms::test::error("error in testToDimShuffle");
  }
}

void testToExpand() {
  Graph g;
  const auto id0 = g.allocate({2, 1, 1}, 0);
  const auto id1 = g.allocate({2, 6, 11}, 0);
  g.allocationToExpand(id0, id1);
  if (!g.areAliased(id0, id1) || !g.containsAliases(id1)) {
    throw poprithms::test::error("error in testToExpand");
  }
}

void testToReverse() {
  Graph g;
  const auto t0 = g.tensor(g.allocate({2}, 0));
  const auto t1 = g.tensor(g.allocate({2}, 0));
  g.allocationToReverse(t0.id(), {0}, t1.id());
  if (!g.areAliased(t0.slice({0}, {1}).id(), t1.slice({1}, {2}).id())) {
    throw poprithms::test::error("error in testToReverse");
  }
}

} // namespace

int main() {
  testToConcat();
  testToSettSample();
  testToDimShuffle();
  testToReshape();
  testToExpand();
  testToReverse();
  return 0;
}
