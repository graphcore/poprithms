// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/schedule/transitiveclosure/error.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace {

using namespace poprithms::schedule::transitiveclosure;

}

int main() {

  /*
   *
   *    0
   *   / \
   *  1   2
   *   \ / \
   *    3   4
   *    |   | \
   *    5   6  7
   *     \  | /
   *       \|/
   *        8
   *
   *        */

  TransitiveClosure em{{{1, 2}, {3}, {3, 4}, {5}, {6, 7}, {8}, {8}, {8}, {}}};

  auto assertSoln = [&em](OpId opId,
                          const std::vector<OpId> &subset,
                          std::tuple<IsFirst, IsFinal> expected) {
    const auto observed = em.getExtremumStatus(opId, subset);
    if (observed != expected) {
      std::ostringstream oss;
      oss << "For OpId opId = " << opId << ", expected " << expected
          << " but observed " << observed << '.';
      throw error(oss.str());
    }
  };

  assertSoln(0, {0, 1, 2}, {IsFirst::Yes, IsFinal::No});
  assertSoln(1, {0, 1, 2}, {IsFirst::No, IsFinal::Maybe});
  assertSoln(1, {0, 1, 2, 3}, {IsFirst::No, IsFinal::No});
  assertSoln(0, {0, 1, 2, 3}, {IsFirst::Yes, IsFinal::No});
  assertSoln(3, {0, 1, 2, 3}, {IsFirst::No, IsFinal::Yes});
  assertSoln(1, {1, 2, 5, 7}, {IsFirst::Maybe, IsFinal::No});
  assertSoln(2, {1, 2, 5, 7}, {IsFirst::Maybe, IsFinal::No});
  assertSoln(5, {1, 2, 5, 7}, {IsFirst::No, IsFinal::Maybe});
  assertSoln(7, {1, 2, 5, 7}, {IsFirst::No, IsFinal::Maybe});
  assertSoln(6, {6}, {IsFirst::Yes, IsFinal::Yes});

  // opId needn't be in subset:
  assertSoln(0, {1, 2}, {IsFirst::Yes, IsFinal::No});
  assertSoln(5, {7, 6, 4}, {IsFirst::Maybe, IsFinal::Maybe});
  assertSoln(5, {8, 7, 6, 4, 1}, {IsFirst::No, IsFinal::No});
  assertSoln(0, {8, 7, 6, 4, 5}, {IsFirst::Yes, IsFinal::No});
  assertSoln(1, {8, 7, 6, 4, 5}, {IsFirst::Maybe, IsFinal::No});

  return 0;
}
