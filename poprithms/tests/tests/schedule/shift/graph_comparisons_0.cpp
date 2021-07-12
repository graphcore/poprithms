// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <string>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>

namespace {

using namespace poprithms::schedule::shift;

void test0() {

  // check multiple comparison operators at the same time
  auto different = [](const Graph &g0, const Graph &g1) {
    return g0 != g1 && !g0.equalTo(g1, false) && ((g0 < g1) != (g1 < g0));
  };

  Graph g0;

  /*
   *
   *  A       B (allocs)
   *  :       :
   *  :       :
   *  a  -->  b (ops)
   *  |
   *  v
   *  c  ==>  d (ops)
   *
   * */

  auto a = g0.insertOp("a");
  auto b = g0.insertOp("b");
  auto c = g0.insertOp("c");
  auto d = g0.insertOp("d");
  g0.insertConstraint(a, b);
  g0.insertConstraint(a, c);
  g0.insertLink(c, d);

  auto A = g0.insertAlloc(100.);
  auto B = g0.insertAlloc(200.);
  g0.insertOpAlloc(a, A);
  g0.insertOpAlloc(b, B);

  // Exact copy:
  {
    const auto g1 = g0;
    if (g0 != g1 || g0 < g1 || !g0.equalTo(g1, false) ||
        g0.lessThan(g1, false)) {
      throw poprithms::test::error("g0 == g1");
    }
  }

  // Extra constraint:
  {
    auto g1 = g0;
    g1.insertConstraint(b, d);
    if (!different(g0, g1)) {
      throw poprithms::test::error(
          "g1 has an extra constraint, not the same");
    }
  }

  // Extra op:
  {
    auto g1 = g0;
    g1.insertOp("extra");
    if (!different(g0, g1)) {
      throw poprithms::test::error("g1 has an extra op, not the same");
    }
  }

  // Extra link:
  {
    auto g1 = g0;
    g1.insertLink(a, b);
    if (!different(g0, g1)) {
      throw poprithms::test::error("g1 has an extra link, not the same");
    }
  }

  // Names differ on 1 op:
  {

    auto g1 = g0;
    g1.insertOp("foo");
    auto g2 = g0;
    g2.insertOp("bar");
    if (g1 == g2) {
      throw poprithms::test::error(
          "g1 and g2 are not equal, their ops don't have the same names");
    }
    if (!g1.equalTo(g2, false)) {
      throw poprithms::test::error(
          "g1 and g2 are equal, if the names of ops are excluded");
    }
  }

  // alloc values differ on 1 alloc
  {
    auto g1 = g0;
    g1.insertAlloc(5);
    auto g2 = g0;
    g2.insertAlloc(6);
    if (!different(g1, g2)) {
      throw poprithms::test::error("g1 and g2 do not have the same allocs");
    }
  }

  // allocs assigned to different ops
  {
    auto g1 = g0;
    {
      auto C = g1.insertAlloc(5);
      g1.insertOpAlloc(c, C);
    }

    auto g2 = g0;
    {
      auto D = g2.insertAlloc(5);
      g2.insertOpAlloc(d, D);
    }

    if (g1 == g2 || g1.equalTo(g2, false)) {
      throw poprithms::test::error(
          "The 2 Graphs are not the same, the final alloc is "
          "assigned to different Ops");
    }

    // Now, add allocs so that isomporphically the graphs are the same, but
    // this comparison doesn't do graph isomorphism (too slow).
    {
      auto C = g2.insertAlloc(5);
      g2.insertOpAlloc(c, C);
      auto D = g1.insertAlloc(5);
      g1.insertOpAlloc(d, D);
      if (!different(g1, g2)) {
        throw poprithms::test::error(
            "They shouldn't compare equal here, as we are not doing "
            "a true graph isomorphism.");
      }
    }
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
