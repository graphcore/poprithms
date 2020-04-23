#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

int main() {

  using namespace poprithms::schedule::pathmatrix;

  //
  //   0
  //  /|\
  // 1 2 3    9
  //  \|/ \  / \
  //   4   6  10\
  //   |   | / \ \
  //   5   7   12 13
  //    \ /     \ /\
  //     8      15 14
  //     |       |
  //     11     /
  //      \    /
  //       \  /
  //        16
  //

  PathMatrix em{{
      {1, 2, 3}, // 0
      {4},       // 1
      {4},       // 2
      {4, 6},    // 3
      {5},       // 4
      {8},       // 5
      {7},       // 6
      {8},       // 7
      {11},      // 8
      {6, 13},   // 9
      {7, 12},   // 10
      {16},      // 11
      {15},      // 12
      {14, 15},  // 13
      {},        // 14
      {16},      // 15
      {}         // 16
  }};

  if (em.nPostPost(0, 0) != 10) {
    throw error("Unexpected number of Ops returned in nPostPost(0,0)");
  }
  if (em.nPostPost(5, 7) != 3) {
    std::cout << em.nPostPost(5, 7);
    throw error("Unexpected number of Ops returned in nPostPost(5,7)");
  }
  if (em.nPostPost(7, 12) != 1) {
    std::cout << em.nPostPost(7, 12);
    throw error("Unexpected number of Ops returned in nPostPost(7,12)");
  }
  if (em.nPostPost(0, 14) != 0) {
    throw error("Unexpected number of Ops returned in nPostPost(0,14)");
  }
  if (em.nPostPost(10, 5) != 3) {
    throw error("Unexpected number of Ops returned in nPostPost(10,5)");
  }

  auto up_4_10 = em.getUnconstrainedPost(4, 10);
  std::sort(up_4_10.begin(), up_4_10.end());
  if (up_4_10 != std::vector<OpId>{7, 12, 15}) {
    throw error("Expected unconstrainedPost for 4,10");
  }

  auto up_2_7 = em.getUnconstrainedPost(2, 7);
  if (up_2_7.size() != 0) {
    throw error("Expected no Ops to be unconstrained w.r.t. 2 and after 7");
  }

  if (em.sameUnconstrained(1, 2)) {
    throw error("Expected different unconstrained sets for Ops 1 and 2");
  }

  if (!em.sameUnconstrained(8, 11)) {
    throw error("Expected same sets for Ops 8 and 11");
  }

  if (em.asEarlyAsAllUnconstrained(3)) {
    throw error("3 cannot be scheduled as early as 9, which is in its "
                "unconstrained set");
  }

  if (!em.asEarlyAsAllUnconstrained(9)) {
    throw error(
        "9 can be scheduled as early as any its unconstrained partners");
  }

  return 0;
}
