// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main() {

  using namespace poprithms::schedule::anneal;

  uint64_t nOps{20};

  Graph g0;
  Graph g1;
  for (uint64_t i = 0; i < nOps; ++i) {
    g0.insertOp("op" + std::to_string(i));
    g1.insertOp("op" + std::to_string(i));
  }

  for (uint64_t i = 0; i < nOps - 1; ++i) {
    // 1,2, 4,5, 7,8...
    if (i % 3 != 0) {
      g0.insertConstraint(i, i + 1);
    }

    // 0,2,4,6...
    if (i % 2 == 0) {
      g1.insertConstraint(i, i + 1);
    }
  }

  // i%3 != 0 and i%2 == 1
  auto diff = g0.constraintDiff(g1);

  decltype(diff) expected(nOps);
  for (uint64_t i = 0; i < nOps - 1; ++i) {
    if (i % 3 != 0 && i % 2 == 1) {
      expected[i].push_back(i + 1);
    }
  }

  if (diff != expected) {
    throw error("Diff is not as expected");
  }

  return 0;
}
