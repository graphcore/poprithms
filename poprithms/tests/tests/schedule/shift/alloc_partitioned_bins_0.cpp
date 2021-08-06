// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <ostream>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::shift;

using Bins = std::vector<std::vector<OpAddress>>;
std::ostream &operator<<(std::ostream &ost, const Bins &adds) {
  for (auto a : adds) {
    ost << "\n       ";
    poprithms::util::append(ost, a);
  }
  return ost;
}

void assertBins(const Bins &observed, Bins expected) {
  for (auto &b : expected) {
    std::sort(b.begin(), b.end());
  }
  if (observed != expected) {
    std::ostringstream oss;
    oss << "\nFailure in assertBins. "
        << "Expected" << expected << "\n, but observed" << observed << '.';
    throw poprithms::test::error(oss.str());
  }
}

/**
 * A     A     A
 * |     |     |
 * a --> b --> c
 *
 * d --> e --> f
 * |     |     |
 * B     B     B
 *
 * */
void test0() {
  Graph g0;
  auto ops = g0.insertOps({"a", "b", "c", "d", "e", "f"});
  auto A   = g0.insertAlloc(14.);
  auto B   = g0.insertAlloc(12.);
  g0.insertConstraints({{ops[0], ops[1]}, {ops[1], ops[2]}});
  g0.insertConstraints({{ops[3], ops[4]}, {ops[4], ops[5]}});
  g0.insertOpAlloc({ops[0], ops[1], ops[2]}, A);
  g0.insertOpAlloc({ops[3], ops[4], ops[5]}, B);

  // See comment in graph.hpp on why we expect this.
  auto out = g0.getAllocPartitionedBins();
  for (auto &x : out) {
    std::sort(x.begin(), x.end());
  }
  std::sort(out.begin(), out.end());
  assertBins(out, Bins({{0, 1, 2}, {3, 4, 5}}));
}

/**
 *
 *  a -> b -> c
 *  |    |    |
 *  A    B    A
 *
 *  */
void test1() {
  Graph g0;
  auto c = g0.insertOp("c");
  auto b = g0.insertOp("b");
  auto a = g0.insertOp("a");
  g0.insertConstraints({{a, b}, {b, c}});
  auto B = g0.insertAlloc(12.);
  auto A = g0.insertAlloc(14.);
  g0.insertOpAlloc(a, A);
  g0.insertOpAlloc(b, B);
  g0.insertOpAlloc(c, A);

  // See comment in graph.hpp on why we expect this.
  auto out = g0.getAllocPartitionedBins();
  for (auto &x : out) {
    std::sort(x.begin(), x.end());
  }
  assertBins(out, Bins({{c, b, a}}));
}

/**
 *            C      C
 *            |      |
 *      +---> d ---> e
 *      |
 *  a --+--- b -> c --> f
 *  |        |    |     |
 * A,D       B    A     D,
 *
 * */
void test2() {

  Graph g;
  const auto a = g.insertOp("a");
  const auto b = g.insertOp("b");
  const auto c = g.insertOp("c");
  const auto d = g.insertOp("d");
  const auto e = g.insertOp("e");
  const auto f = g.insertOp("f");

  const auto A = g.insertAlloc(13.);
  const auto B = g.insertAlloc(12.);
  const auto C = g.insertAlloc(.001);
  const auto D = g.insertAlloc(11.);

  g.insertConstraints({{a, b}, {b, c}, {c, f}, {a, d}, {d, e}});

  g.insertOpAlloc({a, c}, A);
  g.insertOpAlloc(b, B);
  g.insertOpAlloc({d, e}, C);
  g.insertOpAlloc({a, f}, D);

  // See comment in graph.hpp on why we expect this.
  auto out = g.getAllocPartitionedBins();
  for (auto &x : out) {
    std::sort(x.begin(), x.end());
  }
  assertBins(out, Bins({{a, b, c, f}, {d, e}}));
}

void test3() {

  Graph g;

  // 10 allocs:
  std::vector<AllocAddress> allocs;

  // the "magic" numbers (they're not really magic, choose whatever you like):
  constexpr uint64_t nAllocs{10};
  constexpr uint64_t nOps{100};
  constexpr uint64_t firstBridgeOp{57};

  // we create isolated ops, where each op initially has just 1 alloc. There
  // are 10 allocs, so initially there are 10 groups in the partitioning of
  // ops by alloc. We then start adding "bridge" allocs, connecting groups. We
  // do this untile all the ops belong to the same partition.

  for (uint64_t i = 0; i < nAllocs; ++i) {
    allocs.push_back(g.insertAlloc(1.));
  }

  // ops, where op[i] has alloc[i%10] : a partition of the ops into 10 groups.
  for (uint64_t i = 0; i < nOps; ++i) {
    g.insertOp("op" + std::to_string(i));
    g.insertOpAlloc(i, allocs[i % nAllocs]);
  }

  for (uint64_t i = 0; i < nAllocs - 1; ++i) {
    if (g.getAllocPartitioned().size() != nAllocs - i) {
      std::ostringstream oss;
      oss << "At this point, " << i
          << " of the original partitions have been "
          << "connected with a shared alloc. "
          << "We therefore expect " << nAllocs << " - " << i << " = "
          << nAllocs - i << " partitions to remain, not "
          << g.getAllocPartitioned().size() << '.';
      throw poprithms::test::error(oss.str());
    }

    // add an alloc, and connect it to 2 contiguous ops: this merges the
    // groups that these 2 ops belong to:
    auto nxt = g.insertAlloc(1.);
    g.insertOpAlloc({firstBridgeOp + i, firstBridgeOp + i + 1}, nxt);
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
}
