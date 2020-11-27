// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <iostream>
#include <sstream>

#include <poprithms/schedule/supercon/error.hpp>
#include <poprithms/schedule/supercon/graph.hpp>
#include <poprithms/schedule/supercon/logging.hpp>

using namespace poprithms::schedule::supercon;

namespace {

std::ostream &operator<<(std::ostream &out,
                         const std::array<NodeId, 4> &value) {
  out << "[" << value[0] << "," << value[1] << ",";
  out << value[2] << "," << value[3] << "]";
  return out;
}

const std::string assertionErrorPrefix{"Failed in assertCorrectness. "};

void assertCorrectness(const std::string &debugString,
                       const Edges &edges,
                       const Couples &couples) {
  std::cout << "\nIn assertCorrectness, case " << debugString << std::endl;

  auto prefix = assertionErrorPrefix + "This for test with debugString " +
                debugString + ". The error: ";

  auto schedule = getFiloSchedule(edges, couples);

  auto nOps = edges.size();
  if (schedule.size() != nOps) {
    throw error(prefix + "Schedule not of expected size.");
  }

  std::vector<uint64_t> schedIndex(nOps);
  for (uint64_t i = 0; i < nOps; ++i) {
    schedIndex[schedule[i]] = i;
  }

  // Constraints (edges) all satisfied
  for (uint64_t from = 0; from < nOps; ++from) {
    for (auto to : edges[from]) {
      if (schedIndex[from] >= schedIndex[to]) {
        throw error(prefix + " A constraint was not satisfied.");
      }
    }
  }

  for (auto alignedPair : couples) {
    auto a = alignedPair[0];
    auto b = alignedPair[1];
    auto c = alignedPair[2];
    auto d = alignedPair[3];
    if ((schedIndex[a] < schedIndex[b]) != (schedIndex[c] < schedIndex[d])) {
      throw error(prefix + "An alignment pair was not satisfied");
    }
  }
}

void test0() {

  //   0     4
  //  1 2   5 6
  //   3     7

  std::vector<std::vector<NodeId>> edges{
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
  assertCorrectness("test0-b", edges, {Couple({1, 2, 5, 6})});
  assertCorrectness("test0-c", edges, {Couple({1, 2, 6, 5})});
  assertCorrectness("test0-d", edges, {Couple({1, 2, 3, 4})});
  assertCorrectness("test0-e", edges, {Couple({1, 2, 4, 3})});
}

void test1() {

  //       0
  // 1 2 3 4 5 7 8 9
  //       10

  std::vector<std::vector<NodeId>> edges{
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
                    {Couple({2, 1, 3, 4}),
                     Couple({2, 3, 4, 5}),
                     Couple({3, 4, 5, 6}),
                     Couple({4, 5, 6, 7}),
                     Couple({5, 6, 7, 8}),
                     Couple({6, 7, 8, 9})});

  assertCorrectness("test1-c",
                    edges,
                    {Couple({1, 2, 3, 4}),
                     Couple({2, 3, 4, 5}),
                     Couple({3, 4, 5, 6}),
                     Couple({4, 5, 6, 7}),
                     Couple({5, 6, 7, 8}),
                     Couple({6, 7, 8, 9})});

  assertCorrectness(
      "test1-d", edges, {Couple({1, 2, 8, 4}), Couple({4, 8, 6, 5})});
  assertCorrectness(
      "test1-d", edges, {Couple({1, 2, 8, 4}), Couple({4, 8, 5, 6})});

  try {
    assertCorrectness(
        "test1-d",
        edges,
        {Couple({1, 2, 3, 4}), Couple({3, 4, 5, 6}), Couple({5, 6, 2, 1})});
  } catch (const poprithms::error::error &e) {
    log().info("CAUGHT an error as EXPECTED. It was " +
               std::string(e.what()));
  }
}

void assertCoupleConstructor(const bool &&expCanConstruct,
                             const std::array<NodeId, 4> &&in) {
  std::cout << "\nIn assertCoupleConstructor, case " << in << std::endl;
  bool constructed = false;
  try {
    Couple a(in);
    constructed = true;
  } catch (const poprithms::error::error &e) {
    if (!expCanConstruct) {
      log().info("CAUGHT an error as EXPECTED. It was " +
                 std::string(e.what()));
    }
  }

  if (constructed != expCanConstruct) {
    std::stringstream inSs;
    inSs << "Unexpectedly ";
    if (!constructed)
      inSs << "un";
    inSs << "able to construct schedule::supercon::Couple (";
    inSs << in;
    inSs << ")";

    throw error(inSs.str());
  }
}

void testCoupleConstructor() {
  // Valid couples.
  assertCoupleConstructor(true, {1, 2, 3, 4});
  assertCoupleConstructor(true, {1, 2, 1, 4});
  // Invalid couples.
  assertCoupleConstructor(false, {1, 1, 3, 4});
  assertCoupleConstructor(false, {1, 2, 3, 3});
  assertCoupleConstructor(false, {1, 2, 1, 2});
  assertCoupleConstructor(false, {1, 2, 2, 1});
}

void assertCanonicalize(const std::array<NodeId, 4> &&expOut,
                        const std::array<NodeId, 4> &&in) {
  std::cout << "\nIn assertCanonicalize, case " << in << std::endl;
  Couple actOut{in};
  if (expOut[0] != actOut[0] || expOut[1] != actOut[1] ||
      expOut[2] != actOut[2] || expOut[3] != actOut[3]) {
    std::stringstream ss;
    ss << "Expected poprithms::schedule::supercon::Graph::canonicalize(";
    ss << in;
    ss << ") to be ";
    ss << expOut;
    ss << " but observed ";
    ss << actOut;
    throw error(ss.str());
  }
}

void testCanonicalize() {
  // Normal cases.
  assertCanonicalize({1, 2, 3, 4}, {1, 2, 3, 4});
  assertCanonicalize({1, 2, 3, 4}, {2, 1, 4, 3});
  assertCanonicalize({1, 2, 3, 4}, {3, 4, 1, 2});
  assertCanonicalize({1, 2, 3, 4}, {4, 3, 2, 1});
  assertCanonicalize({1, 3, 2, 4}, {3, 1, 4, 2});
  assertCanonicalize({1, 4, 2, 3}, {3, 2, 4, 1});

  // Shared OpId.
  assertCanonicalize({1, 2, 1, 3}, {1, 2, 1, 3});
  assertCanonicalize({1, 2, 1, 3}, {2, 1, 3, 1});
  assertCanonicalize({1, 2, 1, 3}, {1, 3, 1, 2});
  assertCanonicalize({1, 2, 1, 3}, {3, 1, 2, 1});
}

} // namespace

int main() {
  test0();
  test1();
  testCoupleConstructor();
  testCanonicalize();
  return 0;
}
