// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>

//
// Reshapes can shatter contiguous regions:
//
//  ....
//  .11.       .....11.
//  .11.       .11..11.
//  .11.  =>   .11..11.
//  .11.       .11.....
//  .11.
//  .11.
//  ....
//
// We test a variant of this.
//

void test0() {

  using namespace poprithms::memory::unwind;

  Graph g;
  const Shape s0{10, 20};

  const auto v0 = g.sink(s0);
  const auto r0 = g.reshape(v0, {20, 10});

  const DisjointRegions in0(s0, {Region::fromBounds(s0, {2, 4}, {8, 9})});
  const auto observed = g.outRegions(in0, 0, r0.opId(), 0);

  const DisjointRegions expected0(
      {20, 10},
      {Region({20, 10}, {{{{12, 8, 4}, {1, 1, 0}}}, {{{5, 5, 4}}}})});

  if (!expected0.equivalent(observed)) {
    std::ostringstream oss;
    oss << "Unexpected output Region from Reshape: " << observed
        << " != " << expected0;
    throw poprithms::test::error(oss.str());
  }

  if (!g.inRegions(observed, 0, r0.opId(), 0).equivalent(in0)) {
    throw poprithms::test::error(
        "Unexpected result inRegions(outRegions(X)) != X when passed "
        "through reshape");
  }
}
int main() {
  test0();
  return 0;
}
