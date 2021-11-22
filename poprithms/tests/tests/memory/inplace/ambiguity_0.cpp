// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <memory/inplace/ops.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;

void baseTest(const Graph &g, bool expected) {
  if (g.containsAmbiguity().detected() != expected) {
    std::ostringstream oss;
    oss << "In test of inplace::Graph::containsAmbiguity(). "
        << "The graph \n"
        << g << "\nwas expected to " << (expected ? "" : "NOT ")
        << "contain ambiguity. ";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {

  /**
   *      +----> modify!
   *      |
   * x ---+
   *      |
   *      +----> modify!
   *
   * */
  Graph g;
  auto x0 = g.variable({10, 10});
  auto m0 = g.modify(x0);
  auto m1 = g.modify(x0);
  baseTest(g, true);

  // adding a constraint between the modifies should resolve the ambiguity.
  g.constraint(m0, m1);
  baseTest(g, false);
}

void test1() {
  // A chain of modifies is fine (no ambiguity).
  Graph g;
  g.modify(g.modify(g.modify(g.modify(g.variable({6, 7})))));
  baseTest(g, false);
}

void test2() {
  Graph g;
  // parallel chains of modifies is fine (no ambiguity).
  g.modify(g.modify(g.modify(g.modify(g.variable({6, 7})))));
  g.modify(g.modify(g.modify(g.modify(g.variable({6, 7})))));
  baseTest(g, false);
}

void test3() {
  Graph g;
  // parallel chains, on non-overlapping slices:
  auto x0 = g.variable({2, 10});
  g.modify(g.slice(x0, {0, 0}, {1, 10}));
  g.modify(g.slice(x0, {1, 0}, {2, 10}));
  baseTest(g, false);
}

void test4() {
  Graph g;
  // parallel chains, on overlapping slices:
  auto x0 = g.variable({3, 10});
  auto a  = g.modify(g.slice(x0, {0, 0}, {2, 10}));
  auto b  = g.modify(g.slice(x0, {1, 0}, {3, 10}));
  baseTest(g, true);
  g.constraint(b, a);
  baseTest(g, false);
}

void test7() {

  /**
   *
   * a -+
   *    |
   *    +-- agate -- modifies -- d : models add (nothing gets modified)
   *    |
   * b -+
   *    |
   *    +-- agate -- modifies -- e : models add_ (b gets modified)
   *    |
   * c -+
   *
   * */

  Graph g;
  const auto a = g.variable({10, 10});
  const auto b = g.variable({10, 10});
  const auto c = g.variable({10, 10});
  const auto d = g.aliasGate({a, b});
  const auto e = g.aliasGate({b, c}, 0);

  // No modifiers in the graph yet, so impossible to have an ambiguity.
  baseTest(g, false);
  g.modify(d);
  g.modify(e);

  // At this point, we're exactly modelling the compute graph above.
  baseTest(g, true);
}

void test8() {

  /**
   *       +----- view changing stuff ----> modifier
   *       |
   *  a ---+
   *       |
   *       +----- view changing stuff ----> modifier
   *
   * a is (indirectly) modified by both the modifiers. If there is a control
   * dependency (topological constraint) between them, directly, then there is
   * no ambiguity.
   *
   * */

  Graph g;
  const auto a = g.variable({5, 7});

  const auto b = g.slice(a, {0, 0}, {3, 7});
  const auto c = g.slice(a, {2, 0}, {5, 7});

  const auto d = g.reverse(b, Dimensions({0, 1}));
  const auto e = g.dimShuffle(c, {{1, 0}});

  baseTest(g, false);
  const auto f = g.modify(d);

  baseTest(g, false);
  const auto h = g.modify(e);

  baseTest(g, true);

  g.constraint(f, h);
  baseTest(g, false);
}

void test9() {
  /**
   * like test8, except there is an open alias gate before each modifier:
   *
   *                            d               f
   *       +----- slice --- alias gate ----> modifier
   *       |
   *  a ---+
   *       |
   *       +----- slice --- alias gate ----> modifier
   *                            e               h
   */
  Graph g;
  const auto a = g.variable({5, 7});

  const auto b = g.slice(a, {0, 0}, {3, 7});
  const auto c = g.slice(a, {2, 0}, {5, 7});

  const auto d = g.aliasGate({b}, 0);
  const auto e = g.aliasGate({c}, 0);

  baseTest(g, false);
  const auto f = g.modify(d);

  // ambiguity between e and f.
  baseTest(g, true);
  const auto h = g.modify(e);

  // ambiguity between e and f, and between d and h.
  baseTest(g, true);

  g.constraint(f, h);
  baseTest(g, true);

  g.constraint(d, h);

  // there is still an ambiguity between e and f.
  baseTest(g, true);

  g.constraint(e, f);
  baseTest(g, false);
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  test7();
  test8();
  test9();
  return 0;
}
