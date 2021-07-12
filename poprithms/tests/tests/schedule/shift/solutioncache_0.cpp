// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {

void test0() {

  using namespace poprithms::schedule::shift;

  SolutionCache cache;

  Graph g;
  g.insertOp("foo");
  g.insertOp("bar");

  auto g2 = g;

  // First Graph. Cache miss.
  {
    ScheduledGraph sg(std::move(g), Settings(), &cache, &cache);
    if (sg.isFromCache()) {
      throw poprithms::test::error(
          "Cache should be empty here, impossible to have a cache hit");
    }
  }

  // Graph which is identical to first Graph. Cache hit.
  {
    ScheduledGraph sg2(std::move(g2), Settings(), &cache, &cache);
    if (!sg2.isFromCache()) {
      throw poprithms::test::error(
          "Identical Graph to one already in cache, should be cache hit");
    }
  }

  // Graph with new names. Cache hit (names don't matter).
  {
    Graph g3;
    g3.insertOp("goo");
    g3.insertOp("mar");
    ScheduledGraph sg3(std::move(g3), Settings(), &cache, &cache);
    if (!sg3.isFromCache()) {
      throw poprithms::test::error(
          "This Graph is identical (except for Op names) to one in "
          "the cache, should be a cache hit.");
    }
  }

  // Graph with new constraint (edge). Cache miss.
  {
    Graph g4;
    const auto a = g4.insertOp("goo");
    const auto b = g4.insertOp("mar");
    g4.insertConstraint(a, b);
    ScheduledGraph sg4(std::move(g4), Settings(), &cache, &cache);
    if (sg4.isFromCache()) {
      throw poprithms::test::error(
          "This Graph is different to previous Graphs, it has a "
          "constraint. ");
    }
  }

  // Graph with extra Op. Cache miss.
  {
    Graph g4;
    g4.insertOp("goo");
    g4.insertOp("mar");
    g4.insertOp("zee");
    ScheduledGraph sg4(std::move(g4), Settings(), &cache, &cache);
    if (sg4.isFromCache()) {
      throw poprithms::test::error(
          "This Graph is different to previous Graphs, it has a "
          "new Op. ");
    }
  }

  // Original Graph, but with no cache provided to look into.
  {
    Graph g3;
    g3.insertOp("goo");
    g3.insertOp("mar");
    ScheduledGraph sg3(std::move(g3), Settings(), nullptr, nullptr);
    if (sg3.isFromCache()) {
      throw poprithms::test::error(
          "Impossible to have a cache hit when no cache provided!");
    }
  }
}

} // namespace

int main() {
  test0();

  return 0;
}
