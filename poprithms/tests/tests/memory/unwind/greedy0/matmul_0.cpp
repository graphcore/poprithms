// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;

void matmul0() {

  Graph g;

  auto lhs = g.sink({2, 3}, "lhs");
  auto rhs = g.sink({3, 3}, "rhs");
  auto out = g.sink({2, 3}, "out");

  auto lhsSource = g.matMulLhsSource({2, 3}, {3, 3});
  auto rhsSource = g.matMulRhsSource({2, 3}, {3, 3});

  double lhsVal{10};
  double rhsVal{20};
  double outVal{5};

  g.insertValuedPair(lhs, lhsSource, lhsVal);
  g.insertValuedPair(rhs, rhsSource, rhsVal);
  g.insertValuedPair(out, lhs, outVal);

  auto soln = Solution(g);

  if (!g.isMatMulLhsSource(soln.inwardsPaths(lhs).at(0).src().opId())) {
    throw poprithms::test::error("Expected layout of lhs to be lhsSource");
  }

  if (!g.isMatMulRhsSource(soln.inwardsPaths(rhs).at(0).src().opId())) {
    throw poprithms::test::error("Expected layout of rhs to be rhsSource");
  }
}
} // namespace

int main() {
  matmul0();
  return 0;
}
