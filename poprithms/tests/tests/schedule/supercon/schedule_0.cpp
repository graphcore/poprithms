// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/schedule/supercon/error.hpp>
#include <poprithms/schedule/supercon/graph.hpp>
#include <poprithms/schedule/supercon/logging.hpp>

namespace {

const std::string assertionErrorPrefix{"Failed in assertCorrectness. "};

using namespace poprithms::schedule::supercon;

void assertCorrectness(const std::string &debugString,
                       const Edges &edges,
                       const std::vector<std::array<OpId, 4>> &alignedPairs) {
  std::cout << "\nIn assertCorrectness, case " << debugString << std::endl;

  auto prefix = assertionErrorPrefix + "This for test with debugString " +
                debugString + ". The error: ";

  auto schedule = getFiloSchedule(edges, alignedPairs);

  auto nOps = edges.size();
  if (schedule.size() != nOps) {
    throw std::runtime_error(prefix + "Schedule not of expected size.");
  }

  std::vector<uint64_t> schedIndex(nOps);
  for (uint64_t i = 0; i < nOps; ++i) {
    schedIndex[schedule[i]] = i;
  }

  // Constraints (edges) all satisfied
  for (uint64_t from = 0; from < nOps; ++from) {
    for (auto to : edges[from]) {
      if (schedIndex[from] >= schedIndex[to]) {
        throw std::runtime_error(prefix + " A constraint was not satisfied.");
      }
    }
  }

  for (auto alignedPair : alignedPairs) {
    auto a = alignedPair[0];
    auto b = alignedPair[1];
    auto c = alignedPair[2];
    auto d = alignedPair[3];
    if ((schedIndex[a] < schedIndex[b]) != (schedIndex[c] < schedIndex[d])) {
      throw std::runtime_error(prefix +
                               "An alignment pair was not satisfied");
    }
  }
}

void test0() {

  //   0     4
  //  1 2   5 6
  //   3     7

  std::vector<std::vector<OpId>> edges{
      {1, 2}, // 0
      {3},    // 1
      {3},    // 2
      {},     // 3
      {5, 6}, // 4
      {7},    // 5
      {7},    // 6
      {}      // 7
  };

  assertCorrectness("test0-a", edges, {});
  assertCorrectness("test0-b", edges, {{1, 2, 5, 6}});
  assertCorrectness("test0-c", edges, {{1, 2, 6, 5}});
  assertCorrectness("test0-d", edges, {{1, 2, 3, 4}});
  assertCorrectness("test0-e", edges, {{1, 2, 4, 3}});
}

void test1() {

  //       0
  // 1 2 3 4 5 7 8 9
  //       10

  std::vector<std::vector<OpId>> edges{
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, // 0
      {10},
      {10},
      {10},
      {10},
      {10},
      {10},
      {10},
      {10},
      {10},
      {}, // 10
  };

  assertCorrectness("test1-a", edges, {});
  assertCorrectness("test1-b",
                    edges,
                    {{2, 1, 3, 4},
                     {2, 3, 4, 5},
                     {3, 4, 5, 6},
                     {4, 5, 6, 7},
                     {5, 6, 7, 8},
                     {6, 7, 8, 9}});

  assertCorrectness("test1-c",
                    edges,
                    {{1, 2, 3, 4},
                     {2, 3, 4, 5},
                     {3, 4, 5, 6},
                     {4, 5, 6, 7},
                     {5, 6, 7, 8},
                     {6, 7, 8, 9}});

  assertCorrectness("test1-d", edges, {{1, 2, 8, 4}, {4, 8, 6, 5}});
  assertCorrectness("test1-d", edges, {{1, 2, 8, 4}, {4, 8, 5, 6}});

  try {
    assertCorrectness(
        "test1-d", edges, {{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 2, 1}});
  } catch (const poprithms::error::error &e) {
    log().info("CAUGHT an error as EXPECTED. It was " +
               std::string(e.what()));
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
