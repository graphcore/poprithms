// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {

using namespace poprithms::schedule::shift;

/**
 * Schedule the Graph #g0, and write the summary to #iSum. Return the
 * ScheduledGraph.
 * */
ScheduledGraph get(Graph g0, const ISummaryWriter &iSum) {

  return ScheduledGraph::fromCache(
      std::move(g0),
      Settings({KahnTieBreaker::FIFO, {}},
               TransitiveClosureOptimizations::allOn(),
               Settings::defaultRotationTermination(),
               RotationAlgo::RIPPLE,
               1011),
      iSum,
      nullptr,
      nullptr);
}

// Test: Catch an error when the path is not a valid one.
void test0() {
  bool caught{false};
  try {
    FileWriter("non-existent-directory");
  } catch (const poprithms::error::error &e) {
    if (e.code() != poprithms::error::Code(12345)) {
      throw poprithms::test::error("Caught the wrong poprithms error");
    }
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch any poprithms error");
  }
}

// Test: empty-string directory name is valid.
void test1() { FileWriter({}, 0); }

class MockWriter : public ISummaryWriter {
public:
  // Instead of creating folders and actually writing to file, this mock class
  // just records the requests to write Graphs.
  void write(const Graph &g0,
             const Graph &g1,
             double totalTime,
             const std::string &additional) const final {
    (void)totalTime;
    if (mustWrite) {
      g0s.push_back(g0);
      g1s.push_back(g1);
      additionals.push_back(additional);
    }
  }

  bool mightWrite(const Graph & /* fromUser */) const final {
    return mustWrite;
  }

  bool willWrite(const Graph &, /* fromUser */
                 double /* totalTime */) const final {
    return mustWrite;
  }

  MockWriter(bool mustWrite_) : mustWrite(mustWrite_) {}

  void appendLivenessProfile(const ScheduledGraph &) const final {}

  void appendScheduleChange(const ScheduleChange &) const final {}

  void writeInitialSchedule(const std::vector<OpAddress> &) const final {}

  void writeFinalSchedule(const std::vector<OpAddress> &) const final {}

  bool mustWrite;

  // stacks of the requested Graph-writes.
  mutable std::vector<Graph> g0s;
  mutable std::vector<Graph> g1s;
  mutable std::vector<std::string> additionals;
};

void test2() {
  MockWriter m(/** must write = */ true);
  const auto g0 = getRandomGraph(20, 3, 6, 1011);
  const auto g1 = getRandomGraph(40, 3, 6, 1011);
  const auto g2 = getRandomGraph(70, 3, 6, 1011);

  const auto cg0 = get(g0, m);
  const auto cg1 = get(g1, m);
  const auto cg2 = get(g2, m);

  if (m.g0s[2] != g2 || m.g1s[2] != cg2.getGraph()) {
    std::ostringstream oss;
    throw poprithms::test::error(
        "Failed to write the correct graphs in mock test. ");
  }

  if (m.additionals[1].find("Scope") == std::string::npos) {
    throw poprithms::test::error(
        "The summary string doesn't look a time component breakdown");
  }
}

void test3() {
  MockWriter m(/* must write = */ false);
  const auto cg0 = get(getRandomGraph(20, 3, 6, 1011), m);

  if (!m.g0s.empty()) {
    throw poprithms::test::error("MockWriter has mustWrite = false, should "
                                 "have been no calls to 'write'");
  }
}

} // namespace

int main() {

  if (std::getenv("POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY")) {
    throw poprithms::test::error("Bailing from test. Unset all poprithms "
                                 "environment variables first.");
  }

  test0();
  test1();
  test2();
  test3();
  return 0;
}
