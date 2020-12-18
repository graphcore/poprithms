// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

int main() {

  using namespace poprithms::memory::inplace;

  Graph g;
  const Shape s0{10, 20};
  const auto v0 = g.variable(s0);

  const DisjointRegions in0(s0, {Region::fromBounds(s0, {2, 4}, {8, 9})});

  // Check that Reshape works
  const auto r0 = g.reshape(v0, {20, 10});
  const DisjointRegions expected0(
      {20, 10},
      {Region({20, 10}, {{{{12, 8, 4}, {1, 1, 0}}}, {{{5, 5, 4}}}})});

  const auto observed0 = g.outRegions(in0, 0, r0.opId(), 0);

  if (!expected0.equivalent(observed0)) {
    std::ostringstream oss;
    oss << "Unexpected output Region from Reshape: " << observed0
        << " != " << expected0;
    throw error(oss.str());
  }

  if (!g.inRegions(observed0, 0, r0.opId(), 0).equivalent(in0)) {
    throw error("Unexpected result inRegions(outRegions(X)) != X when passed "
                "through reshape");
  }

  return 0;
}
