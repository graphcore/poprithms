// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/util/stringutil.hpp>

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

void test1() {

  Graph g;

  auto alloc0 = g.allocate({1}, Color(7));
  auto alloc1 = g.allocate({1}, Color(1));
  auto alloc2 = g.allocate({1}, Color(2));
  auto alloc3 = g.allocate({1}, Color(8));
  auto alloc4 = g.allocate({1}, Color(9));
  auto alloc5 = g.allocate({1}, Color(8)); // <- repeated color.
  auto alloc6 = g.allocate({1}, Color(1)); // <- repeated color.

  auto c = g.concat(
      {alloc0, alloc1, alloc2, alloc3, alloc4, alloc4, alloc5, alloc6}, 0);

  Colors expected{1, 2, 7, 8, 9};
  if (g.colors(c) != expected) {
    std::ostringstream oss;
    oss << "Expected colors to be unique and in "
        << "ascending order (1,2,7,8,9) not ";
    poprithms::util::append(oss, g.colors(c));
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {

  test0();
  test1();
  return 0;
}
