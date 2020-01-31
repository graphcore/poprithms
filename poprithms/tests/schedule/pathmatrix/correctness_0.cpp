#include <iostream>
#include <sstream>
#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

int main() {

  using namespace poprithms::schedule::pathmatrix;

  //
  //
  // diamond
  //
  //
  PathMatrix em{{{1, 2}, {3}, {3}, {}}};
  if (em.earliest(0) != 0 || em.latest(0) != 0) {
    std::ostringstream oss;
    oss << "Start of diamond range returned : ";
    oss << "[" << em.earliest(0) << ", " << em.latest(0) << "]. ";
    oss << " But start of diamond must be scheduled first.";
    throw error(oss.str());
  }
  if (em.earliest(3) != 3 || em.latest(3) != 3) {
    throw error("End of diamond be registered last");
  }
  for (OpId id : {1, 2}) {
    if (em.earliest(id) != 1 || em.latest(id) != 2) {
      throw error("Edge of diamonds must be scheduled at 1 or 2");
    }
  }

  if (!(em.constrained(0, 1) && em.constrained(0, 2) &&
        em.constrained(0, 3) && em.unconstrained(1, 2) &&
        em.constrained(1, 3) && em.constrained(2, 3))) {
    throw error("incorrect diamond constraints");
  }

  if (em.getFwdRedundant().size() != 0 || em.getBwdRedundant().size() != 0) {
    throw error("there are no redundant edges in this diamond");
  }

  //  stripy diamond
  //
  //        X
  //      /  \
  // (1) X -> X (2)
  //     |    |
  //     |    X
  //      \  /
  //        X
  //
  em = PathMatrix{{{1, 2}, {2, 4}, {3}, {4}, {}}};
  for (uint64_t i = 0; i < 5; ++i) {
    if (em.earliest(i) != i || em.latest(i) != i) {
      throw error("stripy diamond has unique schedule");
    }
  }
  auto fwdRed = em.getFwdRedundant();
  std::sort(fwdRed.begin(), fwdRed.end());
  if (fwdRed.size() != 2 || fwdRed[0] != std::array<OpId, 2>{0, 2} ||
      fwdRed[1] != std::array<OpId, 2>{1, 4}) {
    std::ostringstream oss;
    oss << "Exected 2 specific redundant edges, got:\n";
    for (auto x : fwdRed) {
      oss << "(" << std::get<0>(x) << "," << std::get<1>(x) << ")";
    }
    throw error(oss.str());
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
  em          = PathMatrix(edges);
  fwdRed      = em.getFwdRedundant();
  auto bwdRed = em.getBwdRedundant();
  for (uint64_t i = 0; i < nOps; ++i) {
    for (auto j : edges[i]) {
      std::array<OpId, 2> x{i, j};
      bool found =
          (std::find(fwdRed.cbegin(), fwdRed.cend(), x) != fwdRed.cend());
      if (found == ((j - i) == 1)) {
        throw error("unexpected redundant fwd edge");
      }

      std::array<OpId, 2> y{j, i};
      found = (std::find(bwdRed.cbegin(), bwdRed.cend(), y) != bwdRed.cend());
      if (found == ((j - i) == 1)) {
        throw error("unexpected redundant bwd edge");
      }
    }
  }
  for (uint64_t i = 0; i < nOps; ++i) {
    if (em.earliest(i) != i || em.latest(i) != i) {
      throw error("unique schedule expected in test with redundant edges");
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
      throw error("Parallel chain test of earliest-latest range has failed");
    }
  }
  for (uint64_t i = 0; i < 3; ++i) {
    if (!em.unconstrained(i, 3) || !em.unconstrained(i, 4) ||
        !em.unconstrained(i, 5)) {
      throw error("Expected parallel chains to be unconstrained");
    }
  }
  return 0;
}
