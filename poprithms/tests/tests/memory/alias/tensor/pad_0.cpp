// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

// clang-format off
//
//  id  type         ins         shape  outs         aliases  aliased to
//  --- ------------ ----------- ------ ------------ -------- ----------------------
//  0   Allocate(0)  ()          (5,5)  (3,10)       no       (0,3,6,10,13)
//  1   Allocate(1)  ()          (1,5)  (3)          no       (1,3,6)
//  2   Allocate(1)  ()          (2,5)  (3)          no       (2,3,6)
//  3   Concat       (1,0,2)     (8,5)  (6)          no       (0,1,2,3,6,10,13)
//  4   Allocate(1)  ()          (8,1)  (6)          no       (4,6)
//  5   Allocate(1)  ()          (8,3)  (6)          no       (5,6)
//  6   Concat       (4,3,5)     (8,9)  ()           no       (0,1,2,3,4,5,6,10,13)
//  7   Allocate(2)  ()          ()     (8,9,11,12)  no       (7,9,10,12,13)
//  8   Expand       (7)         (0,5)  (10)         no       ()
//  9   Expand       (7)         (3,5)  (10)         yes      (7,9,10,12,13)
//  10  Concat       (8,0,9)     (8,5)  (13)         yes      (0,3,6,7,9,10,12,13)
//  11  Expand       (7)         (8,0)  (13)         no       ()
//  12  Expand       (7)         (8,4)  (13)         yes      (7,9,10,12,13)
//  13  Concat       (11,10,12)  (8,9)  ()           yes      (0,3,6,7,9,10,12,13)
//
// clang-format on

void test0() {

  Color black{0};
  Color red{1};
  Color white{2};

  Graph g;
  const auto alloc = g.allocate({5, 5}, /* Color = */ black);
  const auto p0    = g.pad(alloc, {1, 1}, {2, 3}, red, BroadcastPadding::No);
  const auto p1 = g.pad(alloc, {0, 0}, {3, 4}, white, BroadcastPadding::Yes);

  if (g.containsAliases(p0)) {
    throw poprithms::test::error("p0 was created with no alias padding");
  }
  if (!g.containsAliases(p1)) {
    throw poprithms::test::error("p1 was created with alias padding");
  }
  if (!g.containsColor(p0, red)) {
    throw poprithms::test::error("p0 was created with red padding");
  }
  if (!(g.shape(p1) == Shape{8, 9})) {
    throw poprithms::test::error("p1 has Shape {8,9}");
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
