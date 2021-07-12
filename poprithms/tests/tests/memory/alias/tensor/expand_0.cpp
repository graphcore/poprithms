// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

// id  type      ins        shape         outs   aliases  aliased to
// --- --------- ---------- ------------- ------ -------- ----------------
// 0   Allocate  ()         ()            (1)    no       (0,1,2,3,6)
// 1   Expand    (0)        (1)           (2)    no       (0,1,2,3,6)
// 2   Expand    (1)        (1,1,1)       (3)    no       (0,1,2,3,6)
// 3   Expand    (2)        (5,4,3,2,1)   (6)    yes      (0,1,2,3,6)
// 4   Allocate  ()         (1,4,3,2,1)   (5,6)  no       (4,5,6)
// 5   Expand    (4)        (4,4,3,2,1)   (6)    yes      (4,5,6)
// 6   Concat    (3,3,4,5)  (15,4,3,2,1)  ()     yes      (0,1,2,3,4,5,6)

void test0() {

  Graph g;
  const auto expa0 = g.tensor(g.allocate({}))
                         .expand({1})
                         .expand({1, 1, 1})
                         .expand({5, 4, 3, 2, 1});

  const auto alloc1 = g.tensor(g.allocate({1, 4, 3, 2, 1}));

  const auto out =
      expa0.concat({expa0, alloc1, alloc1.expand({4, 4, 3, 2, 1})}, 0, 0);

  if (out.shape() != Shape{15, 4, 3, 2, 1}) {
    throw poprithms::test::error("Incorrect Shape determined in test0");
  }
}

void test1() {
  Graph g;
  const auto alloc1 = g.tensor(g.allocate({1}));
  const auto alloc2 = g.tensor(g.allocate({1}));
  const auto cat    = alloc1.concat({alloc2}, 0, 0);
  const auto exp    = cat.expand({4, 3, 2});
  if (!exp.intersectsWith(alloc1) || !exp.intersectsWith(alloc2) ||
      !exp.containsAliases()) {
    throw poprithms::test::error("Failure in test1");
  }
}
} // namespace

int main() {
  test0();
  test1();
  return 0;
}
