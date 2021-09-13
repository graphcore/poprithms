// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <string>

#include <schedule/shift/allocsimplifier.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::schedule::shift;

void testCombineAllocsWithCommonOps0() {

  Graph g;

  const auto a = g.insertOp("a");
  const auto b = g.insertOp("b");
  const auto c = g.insertOp("c");

  const auto A = g.insertAlloc(1.);
  const auto B = g.insertAlloc(2.);
  const auto C = g.insertAlloc(3.);
  const auto D = g.insertAlloc(1. + 3.);

  g.insertOpAlloc({a, c}, A);
  g.insertOpAlloc({a, c}, C);
  g.insertOpAlloc(b, B);

  g.insertConstraint(a, b);
  g.insertConstraint(b, c);

  AllocSimplifier::combineAllocsWithCommonOps(g);

  for (auto op : {a, c}) {
    const auto allocs = g.getOp(op).getAllocs();
    if (allocs.size() != 1 ||
        g.getAlloc(allocs[0]).getWeight() != g.getAlloc(D).getWeight()) {
      throw poprithms::test::error(
          "Failed to combine A and C in combineAllocsWithCommonOps test: " +
          g.getSerializationString());
    }
  }
}

void testCombineAllocsWithCommonOps1() {
  Graph g;
  uint64_t nOps = 10;
  for (uint64_t i = 0; i < nOps; ++i) {
    g.insertOp("op" + std::to_string(i));
  }
  const auto addAllocs = [&g, nOps](uint64_t op0) {
    {
      const std::vector<OpAddress> adds{
          op0, (op0 + 5) % nOps, (op0 + 7) % nOps};
      const auto alloc = g.insertAlloc(1. + op0);
      g.insertOpAlloc(adds, alloc);
    }
    {
      const std::vector<OpAddress> adds{op0, (op0 + 7) % nOps};
      const auto alloc = g.insertAlloc(1. + op0);
      g.insertOpAlloc(adds, alloc);
    }
  };

  for (uint64_t i = 0; i < nOps; ++i) {
    // Add 2 allocations, one associated to 2 ops and one associated to 3.
    // All allocs have distinct associations (sufficient use of prime
    // numbers...)
    addAllocs(i);
  }

  // Add some more allocations. These new allocations all have op associations
  // which have already been inserted above, and so they should be absorbed
  // into previous allocs.
  for (uint64_t j : {2, 4, 7}) {
    for (uint64_t k = 0; k < j; ++k) {
      addAllocs(j);
    }
  }

  AllocSimplifier::combineAllocsWithCommonOps(g);

  for (uint64_t i = 0; i < nOps; ++i) {
    auto n = g.getOp(i).nAllocs();
    if (n != 5) {
      std::ostringstream oss;
      oss << "Expected all Ops to have exactly 5 Allocs associated to them. "
          << "This is not the case for op #" << n;
      throw poprithms::test::error(oss.str());
    }
  }
}

void testDisconnectAllocsWithOneOp() {

  Graph g;
  const auto ops = g.insertOps({"a", "b", "c", "d"});
  const auto A   = g.insertAlloc(1.);
  const auto B   = g.insertAlloc(2.);
  const auto C   = g.insertAlloc(3.);
  g.insertOpAlloc(ops[0], B);
  g.insertOpAlloc(ops, C);

  AllocSimplifier::disconnectAllocsWithOneOp(g);
  if (g.getAlloc(A).nOps() != 0 || g.getAlloc(B).nOps() != 0 ||
      g.getAlloc(C).nOps() != 4) {
    throw poprithms::test::error("Failed to disconnect B from its only Op");
  }

  for (auto op : ops) {
    if (g.getOp(op).nAllocs() != 1) {
      throw poprithms::test::error(
          "All the Ops should only be assocated to C");
    }
  }
}

void testDisconnectAllocsWithZeroWeight() {

  Graph g;
  const auto op    = g.insertOp("a");
  const auto alloc = g.insertAlloc(0.0);
  const auto foo   = g.insertAlloc(1.0);
  const auto jee   = g.insertAlloc(0.0);
  g.insertOpAlloc(op, alloc);
  g.insertOpAlloc(op, foo);
  g.insertOpAlloc(op, jee);
  AllocSimplifier::disconnectAllocsWithZeroWeight(g);
  if (g.getOp(0).nAllocs() != 1) {
    throw poprithms::test::error("Failed to disconnect zero allocations");
  }
}

void testDisconnectInbetweenerAllocs() {

  // The "diamond"
  Graph g;
  const auto ops = g.insertOps({"a", "b", "c", "d"});
  g.insertConstraints({{ops[0], ops[1]},
                       {ops[0], ops[2]},
                       {ops[1], ops[3]},
                       {ops[2], ops[3]}});
  auto A = g.insertAlloc(1.0);
  g.insertOpAlloc(ops, A);
  poprithms::schedule::transitiveclosure::TransitiveClosure tc(
      g.getForwardEdges());

  AllocSimplifier::disconnectInbetweenerAllocs(g, tc);

  if (g.getAlloc(A).getOps() != std::vector<OpAddress>{ops[0], ops[3]}) {
    throw poprithms::test::error(
        "Failed to disconnect the diamond inbetweeners from the alloc");
  }

  if (g.getOp(ops[1]).nAllocs() != 0) {
    throw poprithms::test::error(
        "Failed to disconnect the alloc from the inbetweener");
  }
}

void testDisconnectFixedDurationAllocs() {
  Graph g;

  const auto a = g.insertOp("a");
  const auto b = g.insertOp("b");
  const auto c = g.insertOp("c");

  const auto A = g.insertAlloc(100.);
  const auto B = g.insertAlloc(200.);
  const auto C = g.insertAlloc(200.);

  g.insertConstraint(a, b);
  g.insertConstraint(a, c);

  g.insertOpAlloc(a, A);
  g.insertOpAlloc({a, b}, B);
  g.insertOpAlloc({a, b, c}, C);

  AllocSimplifier::disconnectFixedDurationAllocs(
      g,
      poprithms::schedule::transitiveclosure::TransitiveClosure(
          g.getForwardEdges()));

  if (g.getAlloc(A).nOps() != 0) {
    throw poprithms::test::error(
        "A should be disconnected, with only 1 Op it must be fixed duration");
  }

  if (g.getAlloc(B).nOps() != 2) {
    throw poprithms::test::error(
        "B should remain connected. It's duration could be 2 or 3, depending "
        "on the schedule");
  }

  if (g.getAlloc(C).nOps() != 0) {
    throw poprithms::test::error(
        "C should be disconnected, being associated to all the ops mean that "
        "it's duration is fixed (to the complete schedule)");
  }
}

void testConnectContiguousAllocs0() {

  // A chain of ops, behaving like a sequence of elementwise ops:
  //
  //
  //  A      A,B      B,C     C
  //  |       |       |       |
  //  a ----> b ----> c ----> d
  //
  Graph g;
  std::vector<OpAddress> ops;
  std::vector<AllocAddress> allocs;
  for (uint64_t i = 0; i < 10; ++i) {
    ops.push_back(g.insertOp("op_" + std::to_string(i)));
    if (i > 0) {
      g.insertOpAlloc(ops.back(), allocs.back());
      g.insertConstraint(ops[i - 1], ops[i]);
    }
    allocs.push_back(g.insertAlloc(10.));
    g.insertOpAlloc(ops.back(), allocs.back());
  }

  AllocSimplifier::connectContiguousAllocs(
      g,
      poprithms::schedule::transitiveclosure::TransitiveClosure(
          g.getForwardEdges()));

  // Expect all the ops to be associated to one and the same alloc.
  for (uint64_t i = 0; i < 10; ++i) {
    if (g.getOp(i).nAllocs() != 1) {
      throw poprithms::test::error(
          "Expect all ops to be associated to just 1 alloc after running "
          "connectContiguousAllocs on the chain");
    }
    if (i > 0 && g.getOp(i - 1).getAlloc(0) != g.getOp(i).getAlloc(0)) {
      throw poprithms::test::error(
          "Expect all ops to be associated to the same alloc after running "
          "connectContiguousAllocs on the chain");
    }
  }
}

void testConnectContiguousAllocs1() {
  Graph g;
  auto a = g.insertOp("a");
  auto b = g.insertOp("b");
  g.insertConstraint(b, a);

  auto A  = g.insertAlloc(1.0);
  auto B0 = g.insertAlloc(2.0);
  auto B1 = g.insertAlloc(2.0);
  auto B2 = g.insertAlloc(2.0);

  //    a   <--------  b
  // {B0, B1, B2}     {A, B0}

  g.insertOpAlloc(a, A);
  g.insertOpAlloc(a, B0);

  g.insertOpAlloc(b, B0);
  g.insertOpAlloc(b, B1);
  g.insertOpAlloc(b, B2);

  AllocSimplifier::connectContiguousAllocs(
      g,
      poprithms::schedule::transitiveclosure::TransitiveClosure(
          g.getForwardEdges()));

  // a is the last user of B0, and the first user of B1.
  std::cout << g.getSerializationString() << std::endl;

  // I expect this transform to remove the singleton Allocs. All that should
  // be left is 1 Alloc, of size 2, associated to both ops.

  if (g.getOp(0).nAllocs() != 1 || g.getOp(1).nAllocs() != 1 ||
      g.getOp(0).getAlloc(0) != g.getOp(1).getAlloc(0)) {
    throw poprithms::test::error("Failed to disconnect allocs correctly in "
                                 "the test for connecting contiguous allocs");
  }
}

} // namespace

int main() {

  testCombineAllocsWithCommonOps0();
  testCombineAllocsWithCommonOps1();
  testDisconnectAllocsWithOneOp();
  testDisconnectAllocsWithZeroWeight();
  testDisconnectInbetweenerAllocs();
  testDisconnectFixedDurationAllocs();
  testConnectContiguousAllocs0();
  testConnectContiguousAllocs1();

  return 0;
}
