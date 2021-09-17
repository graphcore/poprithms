// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

//
//         A    .
//        / \   .
//       B0 B1  .
//       |   |  .
//       C0 C1  .
//        \ /   .
//         D    .
//
// alloc0  : A, B0, B1
// alloc10 :    B0, C0
// alloc11 :    B1, C1
// alloc2  : D, C0, C1
//
//
// When can B0 be linked to C0?
//  "    "  B1 "    "    "  C1?
//
// linking logic:
//      lowest delta    highest delta
//      ------------    -------------
// @A : +alloc0,        +alloc0
// @B : -alloc0 +alloc1 +alloc1
// @C : -alloc1,        -alloc1 + alloc2
// @D : -alloc2,        -alloc2
//
// to link, need worst case @C less than or equal best case @B:
//
// -alloc1 + alloc2 <= alloc1 - allo0
//
// i.e. 2*alloc1 >= (alloc2 + alloc0)
//
//

void test(double alloc0, double alloc1, double alloc2) {

  using namespace poprithms::schedule::shift;
  Graph g;
  auto opA  = g.insertOp("A");
  auto opB0 = g.insertOp("B0");
  auto opB1 = g.insertOp("B1");
  auto opC0 = g.insertOp("C0");
  auto opC1 = g.insertOp("C1");
  auto opD  = g.insertOp("D");
  g.insertConstraint(opA, opB0);
  g.insertConstraint(opA, opB1);
  g.insertConstraint(opB0, opC0);
  g.insertConstraint(opB1, opC1);
  g.insertConstraint(opC0, opD);
  g.insertConstraint(opC1, opD);

  auto a0  = g.insertAlloc(alloc0);
  auto a10 = g.insertAlloc(alloc1);
  auto a11 = g.insertAlloc(alloc1);
  auto a2  = g.insertAlloc(alloc2);

  g.insertOpAlloc({opA, opB0, opB1}, a0);
  g.insertOpAlloc({opB0, opC0}, a10);
  g.insertOpAlloc({opB1, opC1}, a11);
  g.insertOpAlloc({opC0, opC1, opD}, a2);

  ScheduledGraph sg(
      std::move(g),
      {KahnTieBreaker::RANDOM, {}},
      TransitiveClosureOptimizations::allOff().withLinkTightDrops(true),
      RotationTermination::preStart());
  auto chainLinks = sg.getGraph().getLinkChains();

  if (2 * alloc1 >= alloc0 + alloc2) {
    if (chainLinks.size() != 2) {
      std::ostringstream oss;
      oss << "2*alloc1 = " << 2 * alloc1
          << ". This is greater than or equal to alloc0 + alloc2 (" << alloc0
          << " + " << alloc2
          << "), and so their should be links from B0 to C1 and B1 to C1";
      throw poprithms::test::error(oss.str());
    }
  } else {
    if (chainLinks.size() != 0) {
      std::ostringstream oss;
      oss << "2*alloc1 = " << 2 * alloc1
          << ". This is less than alloc0 + alloc2 (" << alloc0 << " + "
          << alloc2 << "), and so their should be no links.";
      throw poprithms::test::error(oss.str());
    }
  }
}

int main() {

  test(1, 0.99, 1);
  test(0.99, 0.99, 1);
  test(100., 10., 1000.);
  test(1, 0.99, 0.99);
  test(1.2, 1, 0.8);
  test(1, 1, 1);
  test(0.8, 1, .12);
  test(10., 900., 1000.);
  test(0.8, 1, 1.2);
  test(.12, 1, 0.8);
  test(1, 1.01, 1);
  test(1000., 900., 10.);

  std::mt19937 gen(1015);
  // values drawn from [nOps, 2*nOps)
  std::uniform_int_distribution<> dist(1, 10);
  for (int i = 0; i < 100; ++i) {
    test(dist(gen), dist(gen), dist(gen));
  }

  return 0;
}
