// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {
using namespace poprithms::memory::unwind;
using namespace poprithms::compute;

void test0(double vLinearBias, double vLinearT0) {

  /**
   *
   *      bias === linearForBias     +--t0 === linearForT0
   *        |                        |   |
   *        |                +-------+   |
   *        |                |           |
   *        +--------+-------+        sumLikeReduce
   *                 |                   |
   *               sumLike            biasTarget ( === bias).
   *
   *
   *
   * */

  // the largest. We really want the bias to have the correct layout.
  double vSumLike = 100 + vLinearBias + vLinearT0;

  Graph g;

  // bias
  const auto bias          = g.sink({});
  const auto linearForBias = g.source({});
  g.insertValuedPair(bias, linearForBias, vLinearBias);

  // to
  const auto t0          = g.sink({10, 10});
  const auto linearForT0 = g.source({10, 10});
  g.insertValuedPair(t0, linearForT0, vLinearT0);

  // this insert sumLikeReduce and sumLike. It adds a value pair.
  const auto s = g.sumLike({t0, bias}, 0, vSumLike);

  const auto soln = Solution(g);

  const auto pathsToSinks = soln.barriersToSinks();
  if (pathsToSinks.size() != 2) {
    throw poprithms::test::error(
        "There should be 2 paths (one for each sink)");
  }

  const auto p0 = pathsToSinks[0];
  const auto p1 = pathsToSinks[1];

  if (vLinearBias > vLinearT0) {
    // 1 map bias linearly
    // 2 map T0 linearly
    if (p0.src() != linearForBias || p1.src() != linearForT0) {
      throw poprithms::test::error(
          "Failed in case of vLinearBias > vLinearT0");
    }
  }

  else if (vLinearBias < vLinearT0) {
    if (p0.src() != linearForT0 || !g.isSumLikeReduce(p1.src().opId())) {
      throw poprithms::test::error(
          "Failed in case of vLinearT0 > vLinearBias");
    }
  }

  else {
    throw poprithms::test::error(
        "Invalid test, linear mapping values must be distinct");
  }
}

} // namespace

int main() {
  test0(0.5, 0.7);
  test0(0.7, 0.5);
  return 0;
}
