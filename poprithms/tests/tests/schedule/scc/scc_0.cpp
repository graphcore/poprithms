// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::scc;

std::ostream &operator<<(std::ostream &os, const std::vector<uint64_t> &x) {
  poprithms::util::append(os, x);
  return os;
}

std::ostream &operator<<(std::ostream &os, const SCCs &sccs) {
  os << " [ ";
  for (const auto &scc : sccs) {
    os << scc;
  }
  os << " ] ";
  return os;
}

void test2ElementLoop() {
  auto sccs = getStronglyConnectedComponents({{1}, {0}});
  if (sccs.size() != 1) {
    throw poprithms::test::error("1->0->1 : a single component");
  }
  std::sort(sccs[0].begin(), sccs[0].end());
  if (sccs[0] != SCC{0, 1}) {
    throw poprithms::test::error("incorrect 2 loop elements");
  }
}

void test2SelfLoops() {
  auto sccs = getStronglyConnectedComponents({{0}, {1}});
  if (sccs.size() != 2) {
    throw poprithms::test::error("0->0 and 1->1 : 2 components");
  }
  std::sort(sccs.begin(), sccs.end());
  if (sccs != SCCs{{0}, {1}}) {
    throw poprithms::test::error("incorrect self-loop elements");
  }
}

void testJustADag() {

  // A DAG: nodes only go to nodes with higher indices.
  const auto sccs = getStronglyConnectedComponents({
      {1, 3}, // 0
      {2, 9}, // 1
      {},     // 2
      {4, 9}, // 3
      {8},    // 4
      {6},    // 5
      {8},    // 6
      {8, 9}, // 7
      {},     // 8
      {}      // 9
  });

  if (sccs.size() != 10) {
    throw poprithms::test::error(
        "Just a DAG, should have all singleton components");
  }
}

void test2Loops() {
  const auto sccs = getStronglyConnectedComponents({{1}, {0}, {4}, {2}, {3}});
  if (sccs.size() != 2) {
    throw poprithms::test::error("expected 2 SCCs : {0,1}, {2,3,4}");
  }
}

int count(const std::string &s, const std::string &sub) {
  int n{0};
  auto found = s.find(sub);
  while (found != std::string::npos) {
    ++n;
    found = s.find(sub, found + 1);
  }
  return n;
}

void testSummarySingletonLoop() {

  // 0 -> {0}
  // 1 -> {}
  FwdEdges edges({{0}, {}});
  std::string opName0{"loopyElm"};
  std::string opName1{"looplessElm"};
  const auto summary =
      getSummary(edges, {opName0, opName1}, IncludeCyclelessComponents::No);
  if (count(summary, opName0) != 1) {
    throw poprithms::test::error(
        "Failed to include singleton loopy node in summary");
  }

  if (count(summary, opName1) != 0) {
    throw poprithms::test::error(
        "Incorrectly included singleton loopless node in summary");
  }
}

void testSummary0() {

  /**
   *
   * 2 triangle cycles
   *
   *  a-->b
   *  ^   |
   *  |   v
   *  +---c
   *
   *  d-->e
   *  ^   |
   *  |   v
   *  +---fragilistic
   *
   * */

  FwdEdges edges({{1}, {2}, {0}, {4}, {5}, {3}});

  const auto summary = getSummary(edges,
                                  {"a", "b", "c", "d", "e", "fragilistic"},
                                  IncludeCyclelessComponents::Yes);

  if (count(summary, "in this Strongly Connected Component:  (0->1->2->0)") !=
      2) {
    throw poprithms::test::error(
        "With local co-ordinates, both of the cycles should be 0->1->2->0. "
        "Error message was \n" +
        summary);
  }
}

void assertCycles(const FwdEdges &edges,
                  const std::vector<std::vector<uint64_t>> &expected) {
  const auto cycles = getCycles(getStronglyConnectedComponents(edges), edges);

  // With the current algorithm, this is the expected set of cycles.
  if (cycles != expected) {
    std::ostringstream oss;
    oss << "Cycles not as expected with current algorithm. Expected "
        << expected << " but observed " << cycles
        << ". The current algorithm returns a shortest cycle starting from "
           "the first node in each component. ";
    throw poprithms::test::error(oss.str());
  }

  const auto summary = getSummary(edges,
                                  std::vector<std::string>(edges.size()),
                                  IncludeCyclelessComponents::Yes);

  if (count(summary, "One cycle (out of potentially many)") !=
      std::count_if(expected.cbegin(), expected.cend(), [](const auto &x) {
        return !x.empty();
      })) {
    throw poprithms::test::error(
        "Summary does not report the expected number of cycles");
  };
}

void testCycles0() {

  /**
   *   0->1->3--->2--->4---+
   *   |     |         |   |
   *   +--<--+         +-<-+
   *   |     |
   *   +<-5<-+
   *
   * */

  FwdEdges edges({{1}, {3}, {4}, {0, 2, 5}, {4}, {0}});
  std::vector<std::vector<uint64_t>> expected{{0, 1, 3, 0}, {}, {4, 4}};
  assertCycles(edges, expected);
}

void testCycles1() {
  // The shortest cycle is not found:
  FwdEdges edges({{1}, {2}, {3, 2}, {4, 2}, {5, 3}, {0}});
  assertCycles(edges, {{0, 1, 2, 3, 4, 5, 0}});
}

void testCycles2() {
  FwdEdges edges({{1}, {2, 4}, {0}, {4}, {5}, {3}, {7}, {8, 1}, {6}});
  assertCycles(edges, {{6, 7, 8, 6}, {0, 1, 2, 0}, {3, 4, 5, 3}});
}

void testCycles3() {
  FwdEdges edges({{1}, {2}, {3}, {4}, {5, 2}, {}});
  assertCycles(edges, {{}, {}, {2, 3, 4, 2}, {}});
}

void testDiamond0() {

  // Four strongly connected components, with a super-DAG structure
  // between them.

  //     A
  //     =
  //   10  1       B
  //               =
  //     2        3  5
  //               7
  //    C                     D
  //    =                     =
  //   4  6                  9  0
  //    8                    11 12
  //
  //
  //
  //   A   ->  B
  //
  //   |       |
  //   v       v
  //
  //   C   ->  D
  //

  FwdEdges edges{
      {9, 11, 12}, // 0
      {2},         // 1
      {8, 10},     // 2
      {5},         // 3
      {8},         // 4
      {7, 12},     // 5
      {4, 11},     // 6
      {12, 3},     // 7
      {6},         // 8
      {0, 11},     // 9
      {1, 7},      // 10
      {12, 9},     // 11
      {11, 0}      // 12
  };

  auto components = getStronglyConnectedComponents(edges);
  for (auto &c : components) {
    std::sort(c.begin(), c.end());
  }
  if (components.size() != 4) {
    throw poprithms::test::error("expected 4 components in testDiamond0");
  }

  SCCs expected;

  // first:
  expected.push_back({1, 2, 10});

  // second and third:
  expected.push_back({3, 5, 7});
  expected.push_back({4, 6, 8});

  // final:
  expected.push_back({0, 9, 11, 12});

  if (components[1] == expected[2]) {
    std::swap(expected[1], expected[2]);
  }

  for (uint64_t i = 0; i < 4; ++i) {
    if (components[i] != expected[i]) {
      std::ostringstream oss;
      oss << "Expected \n"
          << expected << ", but observed \n"
          << components << ".";
      throw poprithms::test::error(oss.str());
    }
  }
}

void testPerformance0() {

  uint64_t nOps{100};
  uint64_t maxEdgesPerOp{12};

  std::mt19937 gen(1011);

  FwdEdges edges(nOps);
  for (uint64_t i = 0; i < nOps; ++i) {
    for (uint64_t j = 0; j < i % maxEdgesPerOp; ++j) {
      edges[i].push_back(gen() % nOps);
    }
  }

  auto components = getStronglyConnectedComponents(edges);
  auto summary    = getSummary(edges,
                            std::vector<std::string>(nOps, "x"),
                            IncludeCyclelessComponents::Yes);
}

} // namespace

int main() {
  testDiamond0();
  test2ElementLoop();
  test2SelfLoops();
  testJustADag();
  test2Loops();
  testCycles0();
  testCycles1();
  testCycles2();
  testCycles3();
  testSummary0();
  testPerformance0();
  testSummarySingletonLoop();
  return 0;
}
