// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {

using namespace poprithms::schedule::shift;

// A sort of mock class which has the class being mocked as a member variable.
class TestSolutionCache : public ISolutionCache {

public:
  const std::vector<OpAddress> *find(const Graph &g,
                                     const Settings &s) const final {
    auto x  = sc.find(g, s);
    wasFind = (x != nullptr);
    return x;
  }

  void writeSolution(Graph &&g,
                     const Settings &settings,
                     const std::vector<OpAddress> &soln) final {
    sc.writeSolution(std::move(g), settings, soln);
  }

  mutable bool wasFind{false};

private:
  SolutionCache sc;
};

void test0() {

  TestSolutionCache cache;

  Graph g;
  g.insertOp("foo");
  g.insertOp("bar");

  auto g2 = g;

  // First Graph. Cache miss.
  {
    auto sg = ScheduledGraph::fromCache(
        std::move(g), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.wasFind) {
      throw poprithms::test::error(
          "Cache should be empty here, impossible to have a cache hit");
    }
  }

  // Graph which is identical to first Graph. Cache hit.
  {
    auto sg2 = ScheduledGraph::fromCache(
        std::move(g2), Settings(), FileWriter::Default(), &cache, &cache);
    if (!cache.wasFind) {
      throw poprithms::test::error(
          "Identical Graph to one already in cache, should be cache hit");
    }
  }

  // Graph with new names. Cache hit (names don't matter).
  {
    Graph g3;
    g3.insertOp("goo");
    g3.insertOp("mar");
    auto sg3 = ScheduledGraph::fromCache(
        std::move(g3), Settings(), FileWriter::Default(), &cache, &cache);
    if (!cache.wasFind) {
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
    auto sg00 = ScheduledGraph::fromCache(
        std::move(g4), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.wasFind) {
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
    auto sg00 = ScheduledGraph::fromCache(
        std::move(g4), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.wasFind) {
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
    auto sg00 = ScheduledGraph::fromCache(
        std::move(g3), Settings(), FileWriter::Default(), nullptr, nullptr);
    if (cache.wasFind) {
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
