// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <map>
#include <unordered_map>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/fromcache.hpp>
#include <poprithms/schedule/shift/schedulecache.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::shift;

// TODO(T44953) use gmock.
class TestScheduleCache : public IScheduleCache {

public:
  std::pair<bool, std::vector<OpAddress>>
  findExactStart(const Graph &g, const RotationTermination &rt) const final {
    auto x = sc.findExactStart(g, rt);
    events.push_back("findExactStart:" + std::to_string(x.first));
    return x;
  }

  void writeExactStart(Graph &&g,
                       const RotationTermination &rt,
                       const std::vector<OpAddress> &soln) final {
    events.push_back("writeExactStart");
    sc.writeExactStart(std::move(g), rt, soln);
  }

  // we record all the events that happen to this cache, and then check these
  // events for testing.
  mutable std::vector<std::string> events;

  std::string eventsStr() const {
    std::ostringstream oss;
    oss << "Events : ";
    poprithms::util::append(oss, events);
    return oss.str();
  }

private:
  // The class being tested.
  ScheduleCache sc;
};

void testHotCache() {

  TestScheduleCache cache;

  Graph g;
  g.insertOp("foo");
  g.insertOp("bar");

  std::vector<std::string> expected{"findExactStart:0", "writeExactStart"};

  auto g2 = g;

  // First Graph. Cache miss.
  {
    auto sg = fromCache(
        std::move(g), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.events != expected) {
      throw poprithms::test::error(
          "Cache should be empty here, impossible to have a cache hit. " +
          cache.eventsStr());
    }
  }

  // Graph which is identical to first Graph. Cache hit.
  expected.push_back("findExactStart:1");
  {
    auto sg2 = fromCache(
        std::move(g2), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.events != expected) {
      throw poprithms::test::error(
          "Identical Graph to one already in cache, should be cache hit. " +
          cache.eventsStr());
    }
  }

  // Graph with new names. Cache hit (names don't matter).
  expected.push_back("findExactStart:1");
  {
    Graph g3;
    g3.insertOp("goo");
    g3.insertOp("mar");
    auto sg3 = fromCache(
        std::move(g3), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.events != expected) {
      throw poprithms::test::error(
          "This Graph is identical (except for Op names) to one in "
          "the cache, should be a cache hit. " +
          cache.eventsStr());
    }
  }

  // Graph with new constraint (edge). Cache miss.
  {

    expected.push_back("findExactStart:0");
    expected.push_back("writeExactStart");
    Graph g4;
    const auto a = g4.insertOp("goo");
    const auto b = g4.insertOp("mar");
    g4.insertConstraint(a, b);
    auto sg00 = fromCache(
        std::move(g4), Settings(), FileWriter::Default(), &cache, &cache);

    if (cache.events != expected) {
      throw poprithms::test::error(
          "This Graph is different to previous Graphs, it has a "
          "constraint. " +
          cache.eventsStr());
    }
  }

  // Graph with extra Op. Cache miss.
  {

    expected.push_back("findExactStart:0");
    expected.push_back("writeExactStart");
    Graph g4;
    g4.insertOp("goo");
    g4.insertOp("mar");
    g4.insertOp("zee");
    auto sg00 = fromCache(
        std::move(g4), Settings(), FileWriter::Default(), &cache, &cache);
    if (cache.events != expected) {
      throw poprithms::test::error(
          "This Graph is different to previous Graphs, it has a "
          "new Op. " +
          cache.eventsStr());
    }
  }

  // Original Graph, but with no cache provided to look into.
  {
    Graph g3;
    g3.insertOp("goo");
    g3.insertOp("mar");
    auto sg00 = fromCache(
        std::move(g3), Settings(), FileWriter::Default(), nullptr, nullptr);
    if (cache.events != expected) {
      throw poprithms::test::error(
          "Impossible to have a cache hit when no cache provided!");
    }
  }
}

} // namespace

int main() {
  testHotCache();

  return 0;
}
