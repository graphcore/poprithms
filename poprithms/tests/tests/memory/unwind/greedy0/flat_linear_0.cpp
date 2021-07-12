// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

int main() {

  using namespace poprithms::memory::unwind;

  Graph g;

  // The Tensor we want to find a layout for, In subgraph 0.
  const auto a = g.sink({4, 5});

  // We perform a chain of view-changing operations on Tensor a:
  const auto b = g.dimShuffle(a, {{1, 0}});
  const auto c = g.reverse(b, Dimensions({1}));
  const auto d = g.flatten(c);

  // We know what the desired layout is for the end of the chain of
  // view-changing operations:
  const auto e = g.source({20});
  g.insertValuedPair(e, d, 100.);

  const Solution soln(g);

  // Expectation:
  Chain chain({20});
  chain.reshape({5, 4});
  chain.reverse(Dimension(1));
  chain.dimShuffle({{1, 0}});
  Path p(e, chain.canonicalized(), a);

  auto paths = soln.barriersToSinks();

  if (std::find(paths.cbegin(), paths.cend(), p) == paths.cend()) {
    std::ostringstream oss;
    oss << "Expected the Path " << p << " to appear in the solution";
    throw poprithms::test::error(oss.str());
  }

  if (20 * 100. != soln.getScore()) {
    throw poprithms::test::error("Incorrect score");
  }

  return 0;
}
