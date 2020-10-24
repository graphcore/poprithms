// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <poprithms/schedule/scc/error.hpp>
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
    throw error("1->0->1 : a single component");
  }
  std::sort(sccs[0].begin(), sccs[0].end());
  if (sccs[0] != SCC{0, 1}) {
    throw error("incorrect 2 loop elements");
  }
}

void test2SelfLoops() {
  auto sccs = getStronglyConnectedComponents({{0}, {1}});
  if (sccs.size() != 2) {
    throw error("0->0 and 1->1 : 2 components");
  }
  std::sort(sccs.begin(), sccs.end());
  if (sccs != SCCs{{0}, {1}}) {
    throw error("incorrect self-loop elements");
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
    throw error("Just a DAG, should have all singleton components");
  }
}

void test2Loops() {
  const auto sccs = getStronglyConnectedComponents({{1}, {0}, {4}, {2}, {3}});
  if (sccs.size() != 2) {
    throw error("expected 2 SCCs : {0,1}, {2,3,4}");
  }
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
    throw error("expected 4 components in testDiamond0");
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
      throw error(oss.str());
    }
  }
}

} // namespace

int main() {
  testDiamond0();
  test2ElementLoop();
  test2SelfLoops();
  testJustADag();
  test2Loops();
  return 0;
}
