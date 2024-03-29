// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/util/stringutil.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {

  //  id  type                                 ins  shape    outs  aliases
  //  --- ------------------------------------ ---- -------- ----- --------
  //  0   Allocate                             ()   (50,50)  (1)   no
  //  1   SettSample (((40,10,5))((40,10,5)))  (0)  (40,40)  (2)   no
  //  2   Reshape                              (1)  (1600)   (3)   no
  //  3   Reshape                              (2)  (20,80)  (4)   no
  //  4   Permute (1,0)                        (3)  (80,20)  ()    no
  Graph g;

  const auto arr0 = g.tensor(g.allocate({50, 50}))
                        .slice({5, 5}, {45, 45})
                        .flatten()
                        .reshape({20, 80})
                        .dimShuffle({{1, 0}});

  const auto arr1 = arr0.clone();
  if (arr1.intersectsWith(arr0)) {
    throw poprithms::test::error("Clones should not intersect");
  }
}

void test1() {

  //    id  type      ins      shape    outs  aliases  aliased to
  //    --- --------- -------- -------- ----- -------- ------------
  //    0   Allocate  ()       (10,10)  (3)   no       (0,3,4)
  //    1   Allocate  ()       (10,10)  (3)   no       (1,3,4)
  //    2   Allocate  ()       (10,10)  (3)   no       (2,3,4)
  //    3   Concat    (0,1,2)  (30,10)  (4)   no       (0,1,2,3,4)
  //    4   Reshape   (3)      (5,60)   ()    no       (0,1,2,3,4)
  Graph g;
  const auto arr0 = g.tensor(g.allocate({10, 10}));
  const auto arr1 = g.tensor(g.allocate({10, 10}));
  const auto arr2 = g.tensor(g.allocate({10, 10}));
  const auto cat  = concat({arr0, arr1, arr2}, 0);
  const auto out  = cat.reshape({5, 60});

  //    5   Allocate  ()       (10,10)  (8)   no       (5,8,9)
  //    6   Allocate  ()       (10,10)  (8)   no       (6,8,9)
  //    7   Allocate  ()       (10,10)  (8)   no       (7,8,9)
  //    8   Concat    (5,6,7)  (30,10)  (9)   no       (5,6,7,8,9)
  //    9   Reshape   (8)      (5,60)   ()    no       (5,6,7,8,9)
  const auto outClone = out.clone();

  //    10  Allocate  ()       (10,10)  (0)   no       (10)
  const auto arr0Clone = arr0.clone();

  if (outClone.getNonDisjoint().size() != 5) {
    throw poprithms::test::error(
        "clone has different number of disjoint Tensors");
  }
  if (outClone.intersectsWith(arr0Clone) || outClone.intersectsWith(out)) {
    throw poprithms::test::error(
        "outClone intersects with Tensors in different clone zones");
  }

  if (arr0Clone.getNonDisjoint().size() != 1) {
    throw poprithms::test::error(
        "The clone of an allocation should only intersect with "
        "itself (if no consumers)");
  }
}

void testCloneColorMethod0() {

  Graph g;
  const auto t0 = g.tensor(g.allocate({3, 2}, Color(2)));
  const auto t1 = g.tensor(g.allocate({4, 2}, Color(3)));
  const auto t2 = g.tensor(g.allocate({5, 2}, Color(5)));
  const auto out =
      g.reverse(g.concat({t0.id(), t1.id(), t2.id()}, 0), {0ull, 1ull});
  const auto c0 = g.clone(out, CloneColorMethod::Preserve());
  const auto c1 = g.clone(out, CloneColorMethod::Monochrome(7));

  const auto c0slice0 = g.tensor(c0).reverse({0, 1}).slice({0, 0}, {3, 2});

  if (c0slice0.colors() != Colors{Color(2)}) {
    std::ostringstream oss;
    oss << "This slice of c0 corresponds exactly to t0 in out. "
        << "With clone method Preserve, expect it to have the same "
        << "color as t0 (Color=2). It has colors: ";
    poprithms::util::append(oss, c0slice0.colors());
    throw poprithms::test::error(oss.str());
  }

  if (g.tensor(c0).colors() != Colors{2, 3, 5}) {
    std::ostringstream oss;
    oss << "The tensor c0 is cloned with the Preserve method. "
        << "Expect it therefore to only have Colors={2,3,5}, not ";
    poprithms::util::append(oss, g.tensor(c0).colors());
    throw poprithms::test::error(oss.str());
  }

  if (g.tensor(c1).colors() != Colors{Color(7)}) {
    std::ostringstream oss;
    oss << "The tensor c1 is cloned with the Monochrome method, with "
        << "Color=7. Expect it therefore to only have Color=7, not ";
    poprithms::util::append(oss, g.tensor(c1).colors());
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {

  test0();
  test1();
  testCloneColorMethod0();

  return 0;
}
