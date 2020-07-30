// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {

  Graph g;

  const auto alloc = g.tensor(g.allocate({50}));

  std::vector<Tensor> slices;
  for (int64_t i = 0; i < 5; ++i) {
    slices.push_back(alloc.slice({10 * i}, {10 * (i + 1)}));
  }

  // 12304
  const auto cat0 =
      slices[1].concat({slices[2], slices[3], slices[0], slices[4]}, 0, 0);
  const auto cat1 =
      slices[3].concat({slices[1], slices[2], slices[0], slices[4]}, 2, 0);

  const std::vector<int64_t> order{1, 2, 3, 0, 4};
  for (int64_t i = 0; i < 5; ++i) {
    const auto i_u64  = static_cast<uint64_t>(i);
    const auto slice0 = cat0.slice({10 * i}, {10 * (i + 1)});
    const auto slice1 = cat1.slice({10 * i}, {10 * (i + 1)});
    if (!slice0.intersectsWith(slice1)) {
      throw error("slices are the same, test 1 fails");
    }
    if (!slice0.intersectsWith(slices[order[i_u64]])) {
      throw error("slices are the same, test 2 fails");
    }
    if (slice0.intersectsWith(slices[i_u64]) && i != order[i]) {
      throw error("slices are not the same, test 3 fails");
    }
  }
}

} // namespace

int main() {

  test0();
  return 0;
}
