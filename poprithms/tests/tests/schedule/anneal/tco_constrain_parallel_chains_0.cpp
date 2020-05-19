// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <vector>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {

constexpr uint64_t nOps0{7};

poprithms::schedule::anneal::Graph
graph0(const std::array<double, nOps0> &weights, double w06) {

  using namespace poprithms::schedule::anneal;
  Graph g;
  auto tco = TransitiveClosureOptimizations::allOff()
                 .withConstrainParallelChains()
                 .withMaxIterations(1);

  //
  //     0
  //   /   \
  //  1     \
  //  |      6
  //  3      |
  //  |      |
  //  2      4
  //   \    /
  //    \  /
  //      5
  //

  for (uint64_t i = 0; i < nOps0; ++i) {
    auto op = g.insertOp("Op" + std::to_string(i));
  }
  g.insertConstraint(0, 1);
  g.insertConstraint(1, 3);
  g.insertConstraint(3, 2);
  g.insertConstraint(2, 5);
  g.insertConstraint(0, 6);
  g.insertConstraint(6, 4);
  g.insertConstraint(4, 5);
  for (uint64_t i = 0; i < nOps0; ++i) {
    auto allocId = g.insertAlloc(weights[i]);
    auto ops     = g.getOp(i).getOuts();
    ops.push_back(i);
    g.insertOpAlloc(ops, allocId);
  }

  auto alloc06 = g.insertAlloc(w06);
  g.insertOpAlloc({0, 6}, alloc06);

  // This should have no effect on the optimization
  auto alloc25 = g.insertAlloc(+100);
  g.insertOpAlloc({2, 5}, alloc25);

  g.initialize(KahnTieBreaker::RANDOM, 1011, tco);
  return g;
}

void test0() {

  using namespace poprithms::schedule::anneal;

  std::vector<std::vector<OpAddress>> expected0{
      {1, 6}, {3}, {5}, {2}, {5}, {}, {4}};

  std::vector<std::vector<OpAddress>> expected1{
      {1, 6}, {3, 6}, {5}, {2, 4}, {5}, {}, {4}};

  if (graph0({100, 1, 1, 1, 1, 1, 1}, +10.0).getForwardEdges() != expected0) {
    throw error("Expected no constraints to be inserted when w06 = +10.");
  }
  if (graph0({100, 1, 1, 1, 1, 1, 1}, +100000.0).getForwardEdges() !=
      expected0) {
    throw error("Expected no constraints to be inserted when w06 = +100000.");
  }
  if (graph0({100, 1, 1, 1, 1, 1, 1}, .0).getForwardEdges() != expected1) {
    std::ostringstream oss;
    oss << "Expected certain constraints to be inserted when w06 = 0. "
        << "This relies on corresponding indices in the longer chain "
        << "being lower than those in the shorter chain. "
        << "i.e. that 1 < 6 and 3 < 4.";
    throw error(oss.str());
  }
  if (graph0({100, 1, 1, 1, 1, 1, 10}, 0).getForwardEdges() != expected1) {
    throw error("Expected a larger weight on 6 to not prevent constraints ");
  }
  if (graph0({100, 1, 1, 1, 1, 1, 0.1}, 0).getForwardEdges() != expected0) {
    throw error("Expected a larger weight on 6 to prevent constraints ");
  }
}

void test1() {

  using namespace poprithms::schedule::anneal;
  Graph g;

  //   root
  //   / \
  //  1   2
  //  |   |
  //  3   |
  //   \  |
  //    \ |
  //     tail

  auto root = g.insertOp("root");
  auto op1  = g.insertOp("op1");
  auto op2  = g.insertOp("op2");
  auto op3  = g.insertOp("op3");
  auto tail = g.insertOp("tail");
  g.insertConstraint(root, op1);
  g.insertConstraint(root, op2);
  g.insertConstraint(op1, op3);
  g.insertConstraint(op3, tail);
  g.insertConstraint(op2, tail);

  auto alloc_r1 = g.insertAlloc(10);
  g.insertOpAlloc({root, op1}, alloc_r1);

  // common across chains, so should be ignored
  auto alloc_123 = g.insertAlloc(100);
  g.insertOpAlloc({op1, op2, op3}, alloc_123);

  auto tco = TransitiveClosureOptimizations::allOff()
                 .withConstrainParallelChains()
                 .withMaxIterations(1);
  g.initialize(KahnTieBreaker::RANDOM, 1011, tco);

  auto outs = g.getOp(op1).getOuts();
  std::sort(outs.begin(), outs.end());
  if (outs != std::vector<OpAddress>{2, 3}) {
    throw error("Expected 2 outs from 1 : 2 and 3");
  }
}

} // namespace

int main() {
  test0();
  test1();
}
