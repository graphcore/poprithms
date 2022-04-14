// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

int main() {

  //
  // outer graph sinks. These are the inputs to the graph.
  //   A :  4 x 10. This will be sliced into A0 and A1.
  //   B :  5 x 6
  //   C :  4 x 6
  //
  // inner graph sinks
  //   a : 4 x 5
  //   b : 5 x 6
  //
  // innerGraph(a, b) = matmul(a,b)
  //
  // outerGraph(A, B, C):
  //   X = call(A[:,0:5], B)
  //   Y = call(A[:,5:10], B)
  //   Z = sum(X, Y, C)
  //
  //
  // Diagramatically,
  //
  //   A(4,10)                B(5,6)          C(4,6)
  //    |  |                    | |              |
  //    |  |                    | |              |
  //    |  +- slice -- A0(4,5) -+----------call -+ ----- sum.
  //    |                         |              |
  //    |                         |              |
  //    slice -- A1(4,5) ------- call  ----------+
  //
  //
  //
  // Task: set layouts for a, b, A, B, C, X, Y, Z. Printing the Graph:
  //

  using namespace poprithms::memory::unwind;

  Graph g;

  const auto a       = g.sink({4, 5}, "a (mm-lhs)");
  const auto aSource = g.source({4, 5});
  g.insertValuedPair(aSource, a, 1.5);

  const auto b       = g.sink({5, 6}, "b (mm-rhs)");
  const auto bSource = g.source({5, 6});
  g.insertValuedPair(bSource, b, 2.0);

  // We choose to make the matmul a barrier, although it might be better to
  // make it a fixedPoint if you want to unwind from its output, backwards
  // through the DAG. (see T32143)
  const auto mmOpId = g.barrier({a, b}, {{4, 6}});
  const TensorId mmOut{mmOpId, 0};
  g.setName(mmOpId, "mm");

  const auto A = g.sink({4, 10}, "A");

  const auto A0 = g.slice(A, {0, 0}, {4, 5});
  g.setName(A0.opId(), "A0 (A[:,0:5])");

  const auto A1 = g.slice(A, {0, 5}, {4, 10});
  g.setName(A1.opId(), "A1 (A[:,5:10])");

  const auto B = g.sink({5, 6}, "B");

  const auto C = g.sink({4, 6}, "C");

  const auto X = g.call(TensorIds{A0, B}, TensorIds{a, b}, {mmOut}, 1.)[0];
  g.setName(X.opId(), "X (call out)");

  const auto Y = g.call(TensorIds{A1, B}, TensorIds{a, b}, {mmOut}, 1.)[0];
  g.setName(Y.opId(), "Y (call out)");

  const auto Z = g.sumLike({X, Y, C}, InIndex(0), 1.);
  g.setName(Z.out().opId(), "Z (tri-numpy out)");

  // Solution, manually:
  Paths paths;

  // a, b get the matmul source layouts:
  paths.push_back(g.fullEmpty(aSource, a));
  paths.push_back(g.fullEmpty(bSource, b));

  // A gets the layout from a, on both sidse:
  Chain ca({4, 5});
  ca.settFillInto({0, 0}, {0, 5});
  Chain cb({4, 5});
  cb.settFillInto({0, 5}, {0, 0});
  paths.push_back(Path(a, ca, A));
  paths.push_back(Path(a, cb, A));

  // B gets b's layout:
  paths.push_back(Path(b, Chain({5, 6}), B));

  // X and y get mmOut's layout, copied from the call:
  paths.push_back(g.fullEmpty(mmOut, X));
  paths.push_back(g.fullEmpty(mmOut, Y));

  // And Y then implies C's layout:
  paths.push_back(g.fullEmpty(Y, C));
  const auto soln0 = Solution(g, std::move(paths));
  // g.setPaths(soln0);
  const auto manualScore = soln0.getScore();

  // Try and get the solution automatically:
  auto g1 = g;

  const Solution soln1(std::move(g1));

  if (soln1.getScore() != manualScore) {
    throw poprithms::test::error(
        "Did not correctly compute the score using Greedy0");
  }

  return 0;
}
