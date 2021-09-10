// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <sstream>

#include <testutil/schedule/transitiveclosure/randomedges.hpp>
#include <testutil/schedule/transitiveclosure/transitiveclosurecommandlineoptions.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::transitiveclosure;

template <typename S, typename T> void appendVector(S &s, const T &t) {
  s << '(';
  for (uint64_t i = 0; i < t.size() - 1; ++i) {
    s << t[i] << ',';
  }
  if (!t.empty()) {
    s << t.back();
  }
  s << ')';
}

void test0() {

  /*
   *   0
   *  / \
   * 1   2
   * |   |
   * 3   |
   *  \ /
   *   4
   *   */

  Edges edges{{1, 2}, {3}, {4}, {4}, {}};
  TransitiveClosure pl(edges);
  if (pl.getUnconstrained(0) != std::vector<OpId>{}) {
    std::ostringstream oss;
    oss << "0 is constrained to be before all other Ops, "
        << " not unconstrained with ";
    appendVector(oss, pl.getUnconstrained(0));
    throw poprithms::test::error(oss.str());
  }

  if (pl.getUnconstrained(1) != std::vector<OpId>{2}) {
    std::ostringstream oss;
    oss << "1 is unconstrained only w.r.t. 2, not:(";
    poprithms::util::append(oss, pl.getUnconstrained(1));
    oss << ").";
    throw poprithms::test::error(oss.str());
  }

  if (pl.getUnconstrained(3) != std::vector<OpId>{2}) {
    throw poprithms::test::error("3 is unconstrained only w.r.t. 2");
  }

  if (pl.getUnconstrained(2) != std::vector<OpId>{1, 3}) {
    throw poprithms::test::error("3 is unconstrained only w.r.t. {1,3}");
  }

  if (pl.getUnconstrained(4) != std::vector<OpId>{}) {
    throw poprithms::test::error("4 is constraned to be after all other Ops");
  }
}

void test1() {

  uint64_t N = 700;
  uint64_t E = 4;
  uint64_t D = 50;
  auto pm    = TransitiveClosure(getRandomEdges(N, E, D, 10111));

  for (uint64_t i = 0; i < N; ++i) {
    const auto &unCons  = pm.getUnconstrained(i);
    uint64_t unConIndex = 0;
    for (uint64_t j = 0; j < N; ++j) {
      if (unConIndex < unCons.size() && j == unCons[unConIndex]) {
        if (!pm.unconstrainedInBothDirections(i, j)) {
          throw poprithms::test::error(
              "Disagreement on whether 2 Ops are constrained (in set)");
        }
        ++unConIndex;
      } else {
        if (i != j && pm.unconstrainedInBothDirections(i, j)) {
          throw poprithms::test::error(
              "Disagreement on whether 2 Ops are constrained (not in set)");
        }
      }
    }
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
