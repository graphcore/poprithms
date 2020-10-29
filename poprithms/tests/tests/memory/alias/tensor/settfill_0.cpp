// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <vector>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {

  Graph g;

  // 0000
  // 0000
  // 1212
  // 3333
  // 1212
  // 3333

  const auto alloc0 = g.tensor(g.allocate({2, 4}));
  const auto alloc1 = g.tensor(g.allocate({2, 2}));
  const auto alloc2 = g.tensor(g.allocate({2, 2}));
  const auto alloc3 = g.tensor(g.allocate({2, 4}));

  const auto r0 = Region({{6, 4}}, {{{{2, 4, 0}}}, {{{4, 0, 0}}}});
  const auto r1 = Region({{6, 4}}, {{{{4, 2, 2}, {1, 1, 0}}}, {{{1, 1, 0}}}});
  const auto r2 = Region({{6, 4}}, {{{{4, 2, 2}, {1, 1, 0}}}, {{{1, 1, 1}}}});
  const auto r3 = Region({{6, 4}}, {{{{3, 3, 3}, {1, 1, 0}}}, {{{4, 0, 0}}}});

  const DisjointRegions regions({6, 4}, {r0, r1, r2, r3});

  // Using Graph API;
  const auto filled = g.settfill(
      {alloc0.id(), alloc1.id(), alloc2.id(), alloc3.id()}, regions);

  if (g.allAliases(filled).size() != 5) {
    throw error("filled is aliased to 4 inputs and itself");
  }

  const auto x2 = g.settsample(filled, r2);
  auto aliases  = g.allAliases(x2);
  std::sort(aliases.begin(), aliases.end());
  if (aliases != std::vector<TensorId>{alloc2.id(), filled, x2}) {
    throw error("Expected x2 to be aliased to filled, and alloc2");
  }

  // Using Tensor API;
  const auto filled2 = alloc2.settfill({alloc0, alloc1, alloc3}, 2, regions);

  const auto t0 = g.tensor(filled);
  for (int64_t r = 0; r < 6; ++r) {
    for (int64_t c = 0; c < 4; ++c) {
      const auto sl0 = t0.slice({r, c}, {r + 1, c + 1});
      const auto sl1 = filled2.slice({r, c}, {r + 1, c + 1});
      if (!sl0.intersectsWith(sl1)) {
        std::ostringstream oss;
        oss << "Using different settfill APIs, "
            << "one from Graph and one from Tensor, "
            << "should be identical. ";
        throw error(oss.str());
      }
    }
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
