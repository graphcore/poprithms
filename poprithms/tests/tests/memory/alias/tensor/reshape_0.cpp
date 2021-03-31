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

  const auto alloc0   = g.tensor(g.allocate({300, 100}));
  const auto alloc1   = g.tensor(g.allocate({300, 100}));
  const auto alloc2   = g.tensor(g.allocate({300, 100}));
  const auto cat      = concat({alloc0, alloc1, alloc2}, 1);
  const auto slice    = cat.slice({30, 20}, {280, 270});
  const auto reshaped = slice.reshape({125, 5, 100});
  auto cat2           = concat({reshaped, reshaped}, 1);
  for (auto x : std::vector<Tensor>{alloc0, alloc1, alloc2}) {
    if (!cat2.intersectsWith(x)) {
      throw error("Failed to detect intersection");
    }
  }
  std::cout << g << std::endl;
}

void test1() {

  Graph g;

  const auto alloc0 = g.tensor(g.allocate({10, 2}));
  const auto alloc1 = g.tensor(g.allocate({10, 3}));
  const auto alloc2 = g.tensor(g.allocate({10, 1}));
  // 001112
  // 001112
  // 001112
  // 001112
  // 001112
  // 001112
  // 001112
  // 001112
  // 001112
  // 001112
  const auto cat = concat({alloc0, alloc1, alloc2}, 1);

  // 001112001112001112001112001112
  // 001112001112001112001112001112
  const auto reshaped = cat.reshape({2, 30});

  // 001112001112001112
  // 001112001112001112
  auto sliced = reshaped.slice({0, 0}, {2, 18});

  // select the 0s from sliced, using filter
  // 11....11....11....
  const auto sInter0 = sliced.settSample({{2, 18}, {{{{}}}, {{{2, 4, 0}}}}});

  // select the 1s from sliced
  // ..111...111...111.
  const auto sInter1 = sliced.settSample({{2, 18}, {{{{}}}, {{{3, 3, 2}}}}});

  // select the 2s from sliced
  // .....1.....1.....1
  const auto sInter2 = sliced.settSample({{2, 18}, {{{{}}}, {{{1, 5, 5}}}}});

  const auto allocs = std::vector{alloc0, alloc1, alloc2};
  const auto sInter = std::vector{sInter0, sInter1, sInter2};

  for (uint64_t i = 0; i < 3; ++i) {
    for (uint64_t j = 0; j < 3; ++j) {
      if ((i == j) != (allocs[i].intersectsWith(sInter[j]))) {
        std::ostringstream oss;
        oss << "Failure with i = " << i << " and j = " << j;
        throw error(oss.str());
      }
    }
  }

  std::cout << g.verboseString() << std::endl;
}

} // namespace

int main() {

  test0();
  test1();
  return 0;
}
