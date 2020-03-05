#include <iostream>
#include <poprithms/schedule/dfs/dfs.hpp>
#include <poprithms/schedule/dfs/error.hpp>

int main() {

  //   0 -> 1 -> 2 -> 0
  //        |
  //   3 -> 4 -> 5 -> 3
  //
  //   6 -> 7
  //
  //   8
  //
  //  3 disconnected graphs, one with cycles.
  //
  //  Expected stacks and time-stamps
  //  0
  //  01
  //  012
  //  01  2
  //  014  2
  //  0145  2
  //  01453  2
  //  0145  23
  //  014  235
  //  01  2354
  //  0  23541
  //  . 235410
  //  6  235410
  //  67  235410
  //  6  2354107
  //  . 23541076
  //  8  23541076
  //  .  235410768
  //
  auto final0 = poprithms::schedule::dfs::postOrder({{1},    // <-0
                                                     {2, 4}, // <-1
                                                     {0},    // <-2
                                                     {4},    // <-3
                                                     {5},    // <-4
                                                     {3},    // <-5
                                                     {7},    // <-6
                                                     {6},    // <-7
                                                     {}}     // <-8
  );
  std::vector<uint64_t> order(final0.size());
  for (uint64_t i = 0; i < final0.size(); ++i) {
    order[final0[i]] = i;
  }
  std::vector<uint64_t> expected{2, 3, 5, 4, 1, 0, 7, 6, 8};
  if (order != expected) {
    throw poprithms::schedule::dfs::error(
        "Unexpected post-order traversal in dfs test");
  }
  return 0;
}
