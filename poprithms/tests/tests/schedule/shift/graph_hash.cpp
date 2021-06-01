// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <string>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>

namespace {

using namespace poprithms::schedule::shift;

void test0() {

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
    if (g0.hash(true) != g1.hash(true)) {
      throw error("g0 == g1 but g0.hash(true) != g1.hash(true)");
    }
    if (g0.hash(false) != g1.hash(false)) {
      throw error("g0 == g1 but g0.hash(false) != g1.hash(false)");
    }
  }

  // Extra constraint:
  {
    auto g1 = g0;
    g1.insertConstraint(b, d);
    if (g0.hash(true) == g1.hash(true)) {
      throw error(
          "g1 has an extra constraint, but g0.hash(true) == g1.hash(true)");
    }
    if (g0.hash(false) == g1.hash(false)) {
      throw error(
          "g1 has an extra constraint, but g0.hash(false) == g1.hash(false)");
    }
  }

  // Extra op:
  {
    auto g1 = g0;
    g1.insertOp("extra");
    if (g0.hash(true) == g1.hash(true)) {
      throw error("g1 has an extra op, but g0.hash(true) == g1.hash(true)");
    }
    if (g0.hash(false) == g1.hash(false)) {
      throw error("g1 has an extra op, but g0.hash(false) == g1.hash(false)");
    }
  }

  // Extra link:
  {
    auto g1 = g0;
    g1.insertLink(a, b);
    if (g0.hash(true) == g1.hash(true)) {
      throw error("g1 has an extra link, but g0.hash(true) == g1.hash(true)");
    }
    if (g0.hash(false) == g1.hash(false)) {
      throw error(
          "g1 has an extra link, but g0.hash(false) == g1.hash(false)");
    }
  }

  // Names differ on 1 op:
  {

    auto g1 = g0;
    g1.insertOp("foo");
    auto g2 = g0;
    g2.insertOp("bar");

    if (g1.hash(true) == g2.hash(true)) {
      throw error("g1 and g2 use different names, but g1.hash(true) == "
                  "g2.hash(true)");
    }
    if (g1.hash(false) != g2.hash(false)) {
      throw error("g1 and g2 use different names only, but g1.hash(false) != "
                  "g2.hash(false)");
    }
  }

  // alloc values differ on 1 alloc
  {
    auto g1 = g0;
    g1.insertAlloc(5);
    auto g2 = g0;
    g2.insertAlloc(6);

    if (g1.hash(true) == g2.hash(true)) {
      throw error("g1 and g2 do not have the same allocs, but g1.hash(true) "
                  "== g2.hash(true)");
    }
    if (g1.hash(false) == g2.hash(false)) {
      throw error("g1 and g2 do not have the same allocs, but g1.hash(false) "
                  "== g2.hash(false)");
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

    if (g1.hash(true) == g2.hash(true)) {
      throw error(
          "The 2 Graphs are not the same, the final alloc is "
          "assigned to different Ops, but g1.hash(true) == g2.hash(true)");
    }
    if (g1.hash(false) == g2.hash(false)) {
      throw error(
          "The 2 Graphs are not the same, the final alloc is "
          "assigned to different Ops, but g1.hash(false) == g2.hash(false)");
    }
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
