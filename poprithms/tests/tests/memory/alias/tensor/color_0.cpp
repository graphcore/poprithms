// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {

  Graph g;

  auto alloc  = g.tensor(g.allocate({50}, Color(7)));
  auto sliced = alloc.slice({7}, {17});
  if (sliced.containsColor({5}) || !sliced.containsColor({7})) {
    throw poprithms::test::error("Sliced of the wrong color in test0");
  }

  auto alloc2 = g.tensor(g.allocate({50}, Color(10)));
  auto cat    = sliced.concat({alloc2}, 0, 0);
  if (cat.containsColor({5}) || !cat.containsColor({7}) ||
      !cat.containsColor({10})) {
    throw poprithms::test::error("Sliced of the wrong color in test0");
  }
}

} // namespace

int main() {

  test0();
  return 0;
}
