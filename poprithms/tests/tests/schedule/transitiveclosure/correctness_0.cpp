// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

int main() {

  using namespace poprithms::schedule::transitiveclosure;

  //
  //
  // diamond
  //
  //
  const Edges diamondEdges{{1, 2}, {3}, {3}, {}};
  TransitiveClosure em{diamondEdges};
  if (em.earliest(0) != 0 || em.latest(0) != 0) {
    std::ostringstream oss;
    oss << "Start of diamond range returned : ";
    oss << "[" << em.earliest(0) << ", " << em.latest(0) << "]. ";
    oss << " But start of diamond must be scheduled first.";
    throw poprithms::test::error(oss.str());
  }
  if (em.earliest(3) != 3 || em.latest(3) != 3) {
    throw poprithms::test::error("End of diamond be registered last");
  }
  for (OpId id : {1, 2}) {
    if (em.earliest(id) != 1 || em.latest(id) != 2) {
      throw poprithms::test::error(
          "Edge of diamonds must be scheduled at 1 or 2");
    }
  }

  if (!(em.constrained(0, 1) && em.constrained(0, 2) &&
        em.constrained(0, 3) && em.unconstrainedInBothDirections(1, 2) &&
        em.constrained(1, 3) && em.constrained(2, 3))) {
    throw poprithms::test::error("incorrect diamond constraints");
  }

  if (em.getFlattenedRedundants(diamondEdges).size() != 0) {
    throw poprithms::test::error(
        "there are no redundant edges in this diamond");
  }

  /*
   *  stripy diamond
   *
   *        X
   *      /  \
   * (1) X -> X (2)
   *     |    |
   *     |    X
   *      \  /
   *        X
   *
   * */
  Edges stripyEdges{{{1, 2}, {2, 4}, {3}, {4}, {}}};
  em = TransitiveClosure{stripyEdges};
  for (uint64_t i = 0; i < 5; ++i) {
    if (em.earliest(i) != i || em.latest(i) != i) {
      throw poprithms::test::error("stripy diamond has unique schedule");
    }
    for (uint64_t j = 0; j < 5; ++j) {
      if (j > i) {
        if (!em.constrained(i, j)) {
          throw poprithms::test::error("Expected constrained to be true");
        }
      } else {
        if (em.constrained(i, j)) {
          throw poprithms::test::error("Expected constrained to be false");
        }
      }
    }
  }

  auto fwdRed = em.getFlattenedRedundants(stripyEdges);
  std::sort(fwdRed.begin(), fwdRed.end());
  if (fwdRed.size() != 2 || fwdRed[0] != std::array<OpId, 2>{0, 2} ||
      fwdRed[1] != std::array<OpId, 2>{1, 4}) {
    std::ostringstream oss;
    oss << "Expected 2 specific redundant edges, got:\n";
    for (auto x : fwdRed) {
      oss << "(" << std::get<0>(x) << "," << std::get<1>(x) << ")";
    }
    throw poprithms::test::error(oss.str());
  }

  //
  //
  // unique schedule, with many redundant edges
  //
  //
  uint64_t nOps{10};
  Edges edges(nOps);
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto d : {i + 1, i + 2, i + 3, i + 4, i + 5}) {
      if (d < nOps) {
        edges[i].push_back(d);
      }
    }
  }
  em     = TransitiveClosure(edges);
  fwdRed = em.getFlattenedRedundants(edges);
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto j : edges[i]) {
      std::array<OpId, 2> x{i, j};
      bool found =
          (std::find(fwdRed.cbegin(), fwdRed.cend(), x) != fwdRed.cend());
      if (found == ((j - i) == 1)) {
        throw poprithms::test::error("unexpected redundant fwd edge");
      }
    }
  }
  for (uint64_t i = 0; i < nOps; ++i) {
    if (em.earliest(i) != i || em.latest(i) != i) {
      throw poprithms::test::error(
          "unique schedule expected in test with redundant edges");
    }
  }

  // parallel chains
  //
  // 0    1    2
  // x -> x -> x
  //
  // 3    4    5
  // x -> x -> x
  //
  em = {{{1}, {2}, {}, {4}, {5}, {}}};
  for (uint64_t i = 0; i < 6; ++i) {
    auto expectedEarliest = i % 3;
    auto expectedLatest   = expectedEarliest + 3;
    if (em.earliest(i) != expectedEarliest ||
        em.latest(i) != expectedLatest) {
      throw poprithms::test::error(
          "Parallel chain test of earliest-latest range has failed");
    }
  }
  for (uint64_t i = 0; i < 3; ++i) {
    if (!em.unconstrainedInBothDirections(i, 3) ||
        !em.unconstrainedInBothDirections(i, 4) ||
        !em.unconstrainedInBothDirections(i, 5)) {
      throw poprithms::test::error(
          "Expected parallel chains to be unconstrained");
    }
  }
  return 0;
}
