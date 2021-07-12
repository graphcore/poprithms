// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {
using namespace poprithms::memory::unwind;

void test0() {

  /*


                sink
              /      \
            /          \
           |            |
     subsample (x1)   slice (y1)
         |               |
      flatten (x2)    subsample (y2)
         |               |
      reverse (x3)     flatten (y3)
          \            /
            \        /
              \    /
               numpy
                |
               z0 <=======  source

   The sink Tensor is partitioned into left and right branches:

   010101
   010101
   010101
   010101

   */

  Graph g;

  const auto sink = g.sink({10, 10});

  // left (0) branch:
  const auto x1 = g.subSample(sink, Strides({2, 1}));
  const auto x2 = g.flatten(x1);
  const auto x3 = g.reverse(x2, Dimensions({0}));

  // right (1) branch:
  const auto y1 = g.slice(sink, {1, 0}, {10, 10});
  const auto y2 = g.subSample(y1, Strides({2, 1}));
  const auto y3 = g.flatten(y2);

  const auto z0 = g.sumLike({x3, y3}, InIndex(0), 11.);

  const auto source = g.source({50});
  g.insertValuedPair(z0.out(), source, 10.);

  const Solution soln(std::move(g));
  // const auto soln = g.setPaths(Graph::Algo::Greedy0);
  std::cout << soln.barriersToSinks() << std::endl;

  // Solution paths through left branch
  Chain c0({50});
  c0.reverse(Dimension(0));
  c0.reshape({5, 10});
  c0.settFillInto(Region({10, 10}, {{{{1, 1, 0}}}, {{}}}));
  Path p0(source, c0, sink);

  // Solution paths through right branch
  Chain c1({50});
  c1.reshape({5, 10});
  c1.settFillInto(Region({10, 10}, {{{{1, 1, 1}}}, {{}}}));
  Path p1(source, c1, sink);

  if (soln.barriersToSinks() != Paths{p0, p1} &&
      soln.barriersToSinks() != Paths{p1, p0}) {
    throw poprithms::test::error("Unexpected Paths ");
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
