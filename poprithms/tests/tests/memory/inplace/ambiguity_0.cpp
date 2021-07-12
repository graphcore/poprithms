// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <memory/inplace/ops.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;

void baseTest(const Graph &g, bool expected) {
  if (g.containsAmbiguity() != expected) {
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

class TestGraph : public Graph {
public:
  using Graph::definitelyContainsAmbiguity;
  using Graph::mightContainAmbiguity;
};

void test5() {
  // This is the example in the comment of Graph::containsAmbiguity
  // implementation.
  TestGraph g;

  //
  //       +---> m0 ----> aliasGate(b)
  //       |                  |
  //  a >--+                  |
  //       |                  |
  //       +--->  m1  <-------+
  //

  auto a  = g.variable({10, 10});
  auto m0 = g.modify(a);
  auto m1 = g.modify(a);
  auto b  = g.aliasGate({m0});
  baseTest(g, true);
  g.constraint(b, m1);
  baseTest(g, false);
  if (!g.mightContainAmbiguity()) {
    std::ostringstream oss;
    oss << "Expected the edge from m0 -> aliasGate to be ommited in this "
           "case, and for an ambiguity to be present. ";
    throw poprithms::test::error(oss.str());
  }
  g.constraint(m0, m1);
  if (g.mightContainAmbiguity()) {
    throw poprithms::test::error(
        "With the addition of m0 -> m1, there should not be any "
        "ambiguity, even with the reduced edge map");
  }
}

// A test showing that using the reduced graph as an initial test for
// ambiguity can help significantly:
void test6() {

  // we build a very large and highly aliased graph, with 1e5 Ops:
  uint64_t chainLength{10};
  uint64_t nChains{10000};

  TestGraph g;
  TensorIds ends_;
  for (uint64_t i = 0; i < nChains; ++i) {
    auto x0 = g.variable({5, 6, 7, 8});
    for (uint64_t j = 0; j < chainLength; ++j) {
      x0 = g.modify(x0);
      x0 = g.dimShuffle(x0, {{1, 2, 3, 0}});
    }
    x0 = g.aliasGate({x0});
    ends_.push_back(x0);
  }
  g.concat(ends_, 0);

  // 25 seconds. 75% in creating TransitiveClosures:
  g.definitelyContainsAmbiguity();

  // 7 seconds. 0% of time in creating TransitiveClosures:
  g.containsAmbiguity();
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  test5();

  bool demonstrateThatTheInitialProxyGraphHelps = false;
  if (demonstrateThatTheInitialProxyGraphHelps) {
    test6();
  }
  return 0;
}
