// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

std::ostream &operator<<(std::ostream &os, const Edges &x) {
  int cnt = 0;
  for (const auto &e : x) {
    os << "\n     " << cnt << ":";
    poprithms::util::append(os, e);
    ++cnt;
  }
  return os;
}

void assertDurationBound(const Edges &edges,
                         const OpIds &opIds,
                         const TransitiveClosure::DurationBound &expected) {

  const TransitiveClosure tc(edges);
  const auto observed = tc.getDurationBound(opIds);
  if (observed != expected) {
    std::ostringstream oss;
    oss << "\nFailed in assertDurationBound test. \nFor edges=" << edges
        << "\n, and with opIds=";
    poprithms::util::append(oss, opIds);
    oss << "\n, expected the DurationBound to be " << expected << ", not "
        << observed << ". ";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {

  /*
   *
   *   0
   *  / \
   * 1   2
   * |   |
   * 3   |
   *  \ /
   *   4
   *
   *   */
  Edges edges{{1, 2}, {3}, {4}, {4}, {}};

  // min schedule : 01324
  //                   ==
  // max schedule : 02134
  //                 ====
  assertDurationBound(edges, {2, 4}, {2, 5});

  assertDurationBound(edges, {1, 2, 3}, {3, 4});
  assertDurationBound(edges, {0, 4}, {5, 6});
  assertDurationBound(edges, {3}, {1, 2});
  assertDurationBound(edges, {4}, {1, 2});
  assertDurationBound(edges, {}, {0, 1});
  assertDurationBound(edges, {0, 3}, {3, 5});
  assertDurationBound(edges, {2, 3}, {2, 4});
}

void test1() {

  /*
   *
   *    0
   *    |
   * +--+--+---+
   * 1  2  3   |      9
   * |  |  |   7      |
   * 4  5  6   |      10
   * +--+--+---+
   *       |
   *       |
   *       8
   *
   *   */

  Edges edges(
      {{1, 2, 3, 7}, {4}, {5}, {6}, {8}, {8}, {8}, {8}, {}, {10}, {}});

  // min duration: when ops 9 and 10 are not interwoven with the others.
  // max duration: when ops 9 and 10 ARE interwoven.
  assertDurationBound(edges, {0, 8}, {9, 12});

  // 1,3,6, and 7 could all be contiguous (duration = 4) or they could have
  // everything other than 0 and 8 inbetween them.
  assertDurationBound(edges, {1, 3, 6, 7}, {4, 10});

  // Note that the longest duration for this case is actually 10, so
  // DurationBound(4,11) would be valid solution. But the method
  // getDurationBound does not guarantee strict bounds, and the
  // implementation is expected to return DurationBound(4,12).
  //
  // The specific reason that Duration(4,11) is not returned is that the
  // algorithm cannot determine that either '8' or '10' must come at the end,
  // they cannot both be internal.
  assertDurationBound(edges, {0, 5, 9}, {4, 12});

  assertDurationBound(edges, {10, 9, 8}, {3, 12});
  assertDurationBound(edges, {4, 1, 3, 6}, {4, 10});
  assertDurationBound(edges, {4, 1, 3, 6, 9, 10}, {6, 12});
}

void test2() {
  Edges edges(1000);
  assertDurationBound(edges, {1, 999, 500, 512, 513, 511}, {6, 1001});
}
} // namespace

int main() {
  test0();
  test1();
  test2();
}
