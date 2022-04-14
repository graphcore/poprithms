// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <testutil/memory/unwind/creatorinserter.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/matmulattractions.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;

void matmul0() {

  Graph g;

  auto lhs = g.sink({2, 3}, "lhs");
  auto rhs = g.sink({3, 3}, "rhs");

  double lhsVal{10};
  double rhsVal{20};
  double outVal{5};
  auto mms = growMatmul<poprithms::unwindtoy::MatMulTensorCreatorInserter>(
      {},
      g,
      MatMulAttractions::Default()
          .lhs(lhsVal)
          .rhs(rhsVal)
          .lhsOut(outVal)
          .lhsOut(outVal),
      lhs,
      rhs);

  auto soln = Solution(g);

  if (soln.inwardsPaths(lhs).at(0).src().opId() != mms.lhsSource().opId()) {
    throw poprithms::test::error("Expected layout of lhs to be lhsSource");
  }

  if (soln.inwardsPaths(rhs).at(0).src().opId() != mms.rhsSource().opId()) {
    throw poprithms::test::error("Expected layout of rhs to be rhsSource");
  }
}

} // namespace

int main() {
  matmul0();
  return 0;
}
